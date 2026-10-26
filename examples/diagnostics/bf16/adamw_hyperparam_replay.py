"""Report-only AdamW scalar-rounding isolation over existing sealed captures.

This adds hypotheses to the original audit; it neither edits its evidence nor
changes any optimizer, policy, or threshold. NumPy is not a CUDA kernel oracle.
"""
import argparse
import math
from pathlib import Path

import numpy as np

import adamw_replay_audit as base
from training_parity.common import digest, inside, object_hash, read, require, write_new
from training_parity.plan import load_plan
from training_parity.storage import Stage


VARIANTS = {
    "fp64_with_fp32_hyperparameters": np.float64,
    "fp32_hyperparameters_first": np.float32,
}
ROLES = ("exp_avg", "exp_avg_sq", "parameter", "update",
         "mathematical_update", "mathematical_parameter")


def scalars(step):
    """Non-capturable TE source's float inputs and host bias correction.

    std::pow(float, int) has the double common type in C++11 and later.
    Its result is subtracted from 1 then assigned to a float correction.
    No claim is made about host libm equality for arbitrary steps; this
    diagnostic is restricted to the three captured steps and checked against
    exact Decimal powers in the tests.
    """
    require(type(step) is int and 1 <= step <= 3, "Only the three supported steps are supported")
    b1, b2 = np.float32(.9), np.float32(.999)
    one = np.float32(1)
    return {
        "beta1": b1, "beta2": b2,
        "one_minus_beta1": one - b1,
        "one_minus_beta2": one - b2,
        "bias_correction1": np.float32(1 - math.pow(float(b1), step)),
        "bias_correction2": np.float32(1 - math.pow(float(b2), step)),
        "lr": np.float32(1e-5), "eps": np.float32(1e-8),
    }


def formula_step(parameter, gradient, previous_m, previous_v, step, dtype):
    """Change scalar preparation while retaining the old array operation order.

    In particular c*(g*g) intentionally matches the original audit. TE writes
    (c*g)*g and can fuse multiply-add operations. Neither variant emulates
    those instruction-level choices, rsqrt/division, or final fused subtraction.
    """
    require(dtype in (np.float32, np.float64), "Unsupported formula precision")
    p, g, m0, v0 = [np.asarray(x, dtype=dtype) for x in
                    (parameter, gradient, previous_m, previous_v)]
    require(p.shape == g.shape == m0.shape == v0.shape and p.size, "Replay shapes differ")
    require(all(np.isfinite(x).all() for x in (p, g, m0, v0)), "Nonfinite replay input")
    require(np.all(v0 >= 0), "Negative previous second moment")
    h = {key: dtype(value) for key, value in scalars(step).items()}
    m = h["beta1"] * m0 + h["one_minus_beta1"] * g
    v = h["beta2"] * v0 + h["one_minus_beta2"] * (g * g)
    mhat, vhat = m / h["bias_correction1"], v / h["bias_correction2"]
    mathematical_update = -h["lr"] * (mhat / (np.sqrt(vhat) + h["eps"]))
    mathematical_parameter = p + mathematical_update
    stored_parameter = mathematical_parameter.astype(np.float32)
    result = {
        "exp_avg": m, "exp_avg_sq": v, "parameter": stored_parameter,
        "update": np.subtract(stored_parameter, np.asarray(parameter, dtype=np.float32),
                              dtype=np.float32),
        "mathematical_update": mathematical_update,
        "mathematical_parameter": mathematical_parameter,
    }
    require(all(np.isfinite(x).all() for x in result.values()), "Replay arithmetic overflow")
    return result


def summarize(rows, variant):
    result = {}
    for role in ROLES:
        entries = [(row, row["formulas"][variant][role]) for row in rows]
        worst, metric = max(entries, key=lambda pair: pair[1]["max_abs"])
        relative = [m["relative_l2"] for _, m in entries if m["relative_l2"] is not None]
        result[role] = {
            "tensor_steps": len(entries),
            "elements": sum(m["elements"] for _, m in entries),
            "equal_elements": sum(m["equal_elements"] for _, m in entries),
            "tensor_step_max_abs_quantiles": base.distribution([m["max_abs"] for _, m in entries]),
            "tensor_step_relative_l2_quantiles": base.distribution(relative) if relative else None,
            "worst": {"parameter": worst["parameter"], "step": worst["step"], **metric},
        }
    return result


def stage_report(plan_path, name, old):
    stage = Stage(plan_path, name)
    base.validate_training(stage.plan["suite"]["training"])
    require(stage.plan["steps"] == [0, 1, 2], "Need all three fresh steps")
    require(stage.plan["plan_sha256"] == old["plan_sha256"], "Original audit plan differs")
    require(stage.rank_hashes == old["rank_receipt_sha256"] and stage.files == old["raw_files"],
            "Original audit capture differs")
    catalog = stage.plan["suite"]["description"]["parameters"]
    require(catalog and len(catalog) == old["parameters"],
            "Canonical parameter count differs from the baseline audit")
    expected = {role + "/" + parameter for role in base.ROLES for parameter in catalog}
    observed = {key for key, info in stage.plan["tensors"].items() if info["role"] in base.ROLES}
    require(expected == observed, "Optimizer catalog differs")
    old_rows = {(row["parameter"], row["step"]): row for row in old["rows"]}
    require(set(old_rows) == {(p, s) for p in catalog for s in stage.plan["steps"]},
            "Original audit row coverage differs")
    rows = []
    with base.CanonicalReader(stage) as reader:
        for parameter, info in sorted(catalog.items()):
            values = [{role: reader.tensor(step, role + "/" + parameter) for role in base.ROLES}
                      for step in stage.plan["steps"]]
            for step, captured in enumerate(values):
                previous = values[step - 1] if step else None
                p0 = previous["parameter"] if previous else captured["initial"]
                m0 = previous["exp_avg"] if previous else np.zeros_like(p0)
                v0 = previous["exp_avg_sq"] if previous else np.zeros_like(p0)
                require(all(x.dtype == np.float32 and np.isfinite(x).all()
                            for x in captured.values()), "Need finite FP32 captures")
                require(np.all(captured["exp_avg_sq"] >= 0), "Negative observed second moment")
                invariants = {"captured_delta": base.exact_check(
                    np.subtract(captured["parameter"], captured["initial"], dtype=np.float32),
                    captured["update"])}
                if previous:
                    invariants["master_continuity"] = base.exact_check(p0, captured["initial"])
                formulas = {}
                for variant, dtype in VARIANTS.items():
                    predictions = formula_step(p0, captured["gradient"], m0, v0, step + 1, dtype)
                    formulas[variant] = {
                        role: base.residual(predictions[role], captured[role.removeprefix("mathematical_")])
                        for role in ROLES
                    }
                rows.append({"parameter": parameter, "group": info["group"], "step": step,
                             "invariants": invariants, "formulas": formulas})
    invariant_failures = [
        {"parameter": row["parameter"], "step": row["step"], "check": key}
        for row in rows for key, value in row["invariants"].items() if not value["passed"]
    ]
    summary = {variant: summarize(rows, variant) for variant in VARIANTS}
    contrast = {}
    for variant, old_variant in (("fp64_with_fp32_hyperparameters", "fp64_explicit"),
                                 ("fp32_hyperparameters_first", "fp32_explicit")):
        contrast[variant] = {}
        for role in ROLES:
            old_max = old["formula_summary"][old_variant][role]["tensor_step_max_abs_quantiles"]["max"]
            new_max = summary[variant][role]["tensor_step_max_abs_quantiles"]["max"]
            comparisons = [(r["formulas"][variant][role]["max_abs"],
                            old_rows[(r["parameter"], r["step"])]["formulas"][old_variant][role]["max_abs"])
                           for r in rows]
            contrast[variant][role] = {
                "baseline_formula": old_variant, "old_max_abs": old_max, "new_max_abs": new_max,
                "new_over_old_max_abs": new_max / old_max if old_max else None,
                "tensor_steps_lower": sum(a < b for a, b in comparisons),
                "tensor_steps_equal": sum(a == b for a, b in comparisons),
                "tensor_steps_higher": sum(a > b for a, b in comparisons),
            }
    return {
        "stage": name, "stage_role": old["stage_role"], "plan_sha256": stage.plan["plan_sha256"],
        "rank_receipt_sha256": stage.rank_hashes, "raw_files": stage.files,
        "parameters": len(catalog), "steps": [0, 1, 2], "parameter_steps": len(rows),
        "capture_seals_verified": True, "invariant_failures": invariant_failures,
        "formula_summary": summary, "contrast_to_unchanged_original_audit": contrast, "rows": rows,
    }


def audit(suite_path, baseline_path):
    suite_path, baseline_path = Path(suite_path).resolve(), Path(baseline_path).resolve()
    if suite_path.is_dir():
        suite_path /= "suite.json"
    suite = read(suite_path)
    suite_hash = suite.pop("suite_sha256")
    require(suite_hash == object_hash(suite), "Suite checksum mismatch")
    old = read(baseline_path)
    require(old["status"] == "complete_report_only" and old["exact_invariant_failures"] == 0,
            "Need a completed original audit with no exact invariant failure")
    require(old["suite_sha256"] == suite_hash and old["suite_file_sha256"] == digest(suite_path),
            "Original audit suite differs")
    old_cases = {item["case"]: item for item in old["cases"]}
    require(set(old_cases) == {case["id"] for case in suite["cases"]}, "Case coverage differs")
    cases = []
    for case in suite["cases"]:
        plan_path = inside(suite_path.parent, case["plan"])
        plan = load_plan(plan_path)
        require(plan["plan_sha256"] == case["plan_sha256"], "Case plan differs")
        old_stages = {item["stage"]: item for item in old_cases[case["id"]]["stages"]}
        require(set(old_stages) == set(plan["stages"]), "Stage coverage differs")
        cases.append({"case": case["id"], "stages": [
            stage_report(plan_path, name, old_stages[name]) for name in plan["stages"]]})
    failures = sum(len(stage["invariant_failures"]) for case in cases for stage in case["stages"])
    return {
        "schema_version": 1, "status": "invariant_failure" if failures else "complete_report_only",
        "acceptance_qualified": False, "formula_gates_applied": False,
        "source_sha256": {"audit": digest(Path(__file__)), "original_audit": digest(Path(base.__file__))},
        "original_audit": {"file": baseline_path.name, "sha256": digest(baseline_path)},
        "suite_sha256": suite_hash, "suite_file_sha256": digest(suite_path),
        "scalar_values": {str(step): {key: float(value) for key, value in scalars(step).items()}
                          for step in (1, 2, 3)},
        "formula_conditioning": "Each side's own gradient, previous captured master/moments; fresh zero moments at step 0",
        "variant_scope": "Change hyperparameter preparation only, keeping the old audit array operation order. FP64 isolates scalar rounding; FP32 additionally has explicit non-fused tensor rounding.",
        "limitations": [
            "Not a TE bitwise oracle: no CUDA FMA, source left-associated second-moment multiply, approximate math, or fused master subtraction emulation",
            "Pinned TE source supports this arithmetic hypothesis; runtime package version does not prove binary identity or actual optimizer class",
            "No residual threshold or end-to-end qualification; original audit and policy are unchanged",
            "Live GPU runtime-only input/source artifacts are not reverified on this controller",
        ],
        "cases": cases, "exact_invariant_failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Output already exists")
    report = audit(args.suite, args.baseline_audit)
    write_new(args.output, report)
    print(f'{report["status"]}: {args.output}; acceptance_qualified=false')
    return 2 if report["exact_invariant_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
