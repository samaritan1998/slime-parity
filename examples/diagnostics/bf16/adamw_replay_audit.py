"""Independent, report-only AdamW replay over sealed canonical captures.

NumPy formulas never call torch.optim, Slime, Megatron, or TE. The library's
reader verifies canonical ownership, rank/process seals and raw checksums; it
does not supply the optimizer expectation. Formula residuals are descriptive:
no tolerance is inferred from reference/candidate disagreement or these data.
Only exact captured delta/continuity identities are asserted automatically.
"""
from contextlib import ExitStack
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np

from training_parity.common import EvidenceError, digest, inside, object_hash, read, require, write_new
from training_parity.plan import load_plan, verify_artifacts
from training_parity.storage import Stage, DTYPES


ROLES = ("initial", "gradient", "exp_avg", "exp_avg_sq", "update", "parameter")
FORMULAS = ("fp64_explicit", "fp32_explicit")
QUANTILES = {"p50": .5, "p90": .9, "p99": .99, "p99_9": .999, "max": 1.0}
TARGET_OPTIMIZER = {"name": "adamw", "lr": 1e-5, "betas": [.9, .999],
                    "eps": 1e-8, "weight_decay": 0}


def validate_training(training):
    require(training.get("optimizer") == TARGET_OPTIMIZER,
            "This audit is scoped to AdamW lr=1e-5, betas=.9/.999, eps=1e-8, weight_decay=0")
    require(training.get("clip_grad") == 0, "Pre-clip capture needs zero clipping for this audit")
    require(type(training.get("steps")) is int and training["steps"] >= 2,
            "Need consecutive fresh optimizer steps")


def formula_step(parameter, gradient, previous_m, previous_v, step, dtype):
    """One step conditioned on this side's observed previous state.

    step is one-based. Each elementary NumPy operation rounds to dtype; there
    is no fused multiply-add, torch lerp, or GPU rsqrt approximation. The FP64
    path is a high-precision equation, not an emulation of a specific kernel.
    The FP32 path is one explicit evaluation order, not a bitwise GPU oracle.
    """
    require(type(step) is int and step >= 1, "Adam step must be positive")
    require(dtype in (np.float32, np.float64), "Unsupported formula precision")
    p, g, m0, v0 = [np.asarray(x, dtype=dtype) for x in
                    (parameter, gradient, previous_m, previous_v)]
    require(p.shape == g.shape == m0.shape == v0.shape, "Replay shapes differ")
    require(all(np.isfinite(x).all() for x in (p, g, m0, v0)), "Nonfinite replay input")
    require(np.all(v0 >= 0), "Negative previous second moment")
    beta1, beta2 = .9, .999
    # Complements and bias corrections are mathematical hyperparameters,
    # computed in Python double then explicitly rounded to formula dtype.
    m = dtype(beta1) * m0 + dtype(1 - beta1) * g
    v = dtype(beta2) * v0 + dtype(1 - beta2) * (g * g)
    mhat = m / dtype(1 - beta1 ** step)
    vhat = v / dtype(1 - beta2 ** step)
    mathematical_update = -dtype(1e-5) * (mhat / (np.sqrt(vhat) + dtype(1e-8)))
    mathematical_parameter = p + mathematical_update
    stored_parameter = mathematical_parameter.astype(np.float32)
    # Capture records the actual stored master difference, not the optimizer's
    # pre-addition mathematical delta. Keep both observations in the report.
    captured_update = np.subtract(stored_parameter, np.asarray(parameter, dtype=np.float32),
                                  dtype=np.float32)
    result = {"exp_avg": m, "exp_avg_sq": v, "parameter": stored_parameter,
              "update": captured_update, "mathematical_update": mathematical_update,
              "mathematical_parameter": mathematical_parameter, "mhat": mhat}
    require(all(np.isfinite(x).all() for x in result.values()), "Replay arithmetic overflow")
    return result


def distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    require(a.size and np.isfinite(a).all(), "Invalid distribution")
    return {name: float(np.quantile(a, q)) for name, q in QUANTILES.items()}


def l2(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(a)))
    return scale * float(np.sqrt(np.sum((a / scale) ** 2))) if scale else 0.0


def residual(expected, actual):
    x, y = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    require(x.shape == y.shape and x.size, "Residual shapes differ")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite residual input")
    difference = y - x
    absolute = np.abs(difference)
    flat = int(np.argmax(absolute))
    reference_norm, error_norm = l2(x), l2(difference)
    return {"elements": x.size, "equal_elements": int(np.count_nonzero(x == y)),
            "max_abs": float(absolute.flat[flat]), "absolute_l2": error_norm,
            "relative_l2": error_norm / reference_norm if reference_norm else (0.0 if not error_norm else None),
            "rmse": error_norm / np.sqrt(x.size), "element_abs_error_quantiles": distribution(absolute),
            "max_abs_location": {"flat_index": flat,
                                 "index": list(map(int, np.unravel_index(flat, x.shape))),
                                 "expected": float(x.flat[flat]), "observed": float(y.flat[flat])}}


def exact_check(expected, actual):
    result = residual(expected, actual)
    return {"passed": bool(np.array_equal(expected, actual)), **result}


def audit_tensor(captures):
    """Audit all fresh steps of one parameter, keeping no cross-side baseline."""
    require(len(captures) >= 2, "At least two captured steps are required")
    shape = captures[0]["initial"].shape
    rows = []
    for index, values in enumerate(captures):
        require(set(values) == set(ROLES), "Incomplete optimizer roles")
        require(all(x.dtype == np.float32 and x.shape == shape and np.isfinite(x).all()
                    for x in values.values()), "Need finite FP32 canonical master/gradient/state captures")
        require(np.all(values["exp_avg_sq"] >= 0), "Negative observed second moment")
        previous = captures[index - 1] if index else None
        p0 = previous["parameter"] if previous else values["initial"]
        m0 = previous["exp_avg"] if previous else np.zeros(shape, dtype=np.float32)
        v0 = previous["exp_avg_sq"] if previous else np.zeros(shape, dtype=np.float32)
        invariants = {"captured_delta": exact_check(
            np.subtract(values["parameter"], values["initial"], dtype=np.float32), values["update"])}
        if previous:
            invariants["master_continuity"] = exact_check(previous["parameter"], values["initial"])
        formulas = {}
        predictions = {}
        for label, dtype in zip(FORMULAS, (np.float64, np.float32)):
            prediction = formula_step(p0, values["gradient"], m0, v0, index + 1, dtype)
            predictions[label] = prediction
            formulas[label] = {role: residual(prediction[role], values[role])
                               for role in ("exp_avg", "exp_avg_sq", "parameter", "update")}
            for role, captured in (("mathematical_update", "update"),
                                   ("mathematical_parameter", "parameter")):
                formulas[label][role] = residual(prediction[role], values[captured])
        g, update = values["gradient"].astype(np.float64), values["update"].astype(np.float64)
        momentum = predictions["fp64_explicit"]["mhat"]
        diagnostics = {"gradient_zero": int(np.count_nonzero(g == 0)),
                       "gradient_nonzero_abs_le_epsilon": int(np.count_nonzero((g != 0) & (np.abs(g) <= 1e-8))),
                       "gradient_abs_gt_epsilon": int(np.count_nonzero(np.abs(g) > 1e-8)),
                       "replayed_mhat_zero": int(np.count_nonzero(momentum == 0)),
                       "replayed_mhat_nonzero_abs_le_epsilon": int(np.count_nonzero((momentum != 0) & (np.abs(momentum) <= 1e-8))),
                       "captured_update_zero": int(np.count_nonzero(update == 0)),
                       "update_same_sign_as_replayed_mhat": int(np.count_nonzero(update * momentum > 0)),
                       "near_zero_definition": "abs(value)<=optimizer epsilon; not a model-scale tolerance or exemption"}
        rows.append({"step": index, "invariants": invariants, "formulas": formulas,
                     "diagnostics": diagnostics})
    return rows


class CanonicalReader:
    """Read already-verified pieces once per file, without modifying evidence."""
    def __init__(self, stage):
        self.stage, self.stack, self.streams = stage, ExitStack(), {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def tensor(self, step, name):
        pieces = self.stage.indices[(step, name)]
        require(all(p["dtype"] == "float32" for p in pieces), "This audit requires FP32 stored optimizer tensors")
        shape = self.stage.plan["tensors"][name]["shape"]
        result = np.empty(int(np.prod(shape)), dtype=np.float32)
        for piece in pieces:
            file = piece["file"]
            if file not in self.streams:
                self.streams[file] = self.stack.enter_context(inside(self.stage.folder, file).open("rb"))
            stream = self.streams[file]
            stream.seek(piece["offset_bytes"])
            size = piece["elements"] * DTYPES["float32"].itemsize
            raw = stream.read(size)
            require(len(raw) == size, "Truncated canonical capture during replay")
            result[piece["start"]:piece["end"]] = np.frombuffer(raw, dtype=DTYPES["float32"])
        return result.reshape(shape)


def summarize(rows):
    output = {}
    for variant in FORMULAS:
        output[variant] = {}
        for role in rows[0]["formulas"][variant]:
            items = [(r, r["formulas"][variant][role]) for r in rows]
            worst, metrics = max(items, key=lambda item: item[1]["max_abs"])
            output[variant][role] = {
                "tensor_steps": len(items), "elements": sum(m["elements"] for _, m in items),
                "equal_elements": sum(m["equal_elements"] for _, m in items),
                "tensor_step_max_abs_quantiles": distribution([m["max_abs"] for _, m in items]),
                "tensor_step_relative_l2_quantiles": distribution([m["relative_l2"] for _, m in items if m["relative_l2"] is not None])
                    if any(m["relative_l2"] is not None for _, m in items) else None,
                "undefined_relative_l2_tensor_steps": sum(m["relative_l2"] is None for _, m in items),
                "worst": {"parameter": worst["parameter"], "step": worst["step"], **metrics}}
    return output


def audit_stage(plan_path, name, expected_parameters=None, expected_steps=None):
    stage = Stage(plan_path, name)
    plan = stage.plan
    training = plan["suite"]["training"]
    validate_training(training)
    require(plan["steps"] == list(range(training["steps"])), "Need every fresh step starting at zero")
    if expected_steps is not None:
        require(len(plan["steps"]) == expected_steps, "Unexpected optimizer step count")
    catalog = plan["suite"]["description"]["parameters"]
    if expected_parameters is not None:
        require(len(catalog) == expected_parameters, "Unexpected canonical parameter count")
    required = {role + "/" + parameter for parameter in catalog for role in ROLES}
    observed_roles = {n for n, tensor in plan["tensors"].items() if tensor["role"] in ROLES}
    require(required == observed_roles, "Canonical optimizer catalog is missing or has extra parameters")
    rows = []
    with CanonicalReader(stage) as reader:
        for parameter, description in sorted(catalog.items()):
            for role in ROLES:
                tensor = plan["tensors"][role + "/" + parameter]
                require(tensor["shape"] == description["shape"] and tensor["role"] == role,
                        "Canonical shape/role differs from parameter contract")
            captures = [{role: reader.tensor(step, role + "/" + parameter) for role in ROLES}
                        for step in plan["steps"]]
            for row in audit_tensor(captures):
                row.update(parameter=parameter, group=description["group"], shape=description["shape"])
                row["canonical_piece_counts"] = {role: len(stage.indices[(row["step"], role + "/" + parameter)])
                                                 for role in ROLES}
                rows.append(row)
    invariant_failures = [{"parameter": r["parameter"], "step": r["step"], "check": check,
                           "max_abs": result["max_abs"]} for r in rows
                          for check, result in r["invariants"].items() if not result["passed"]]
    return {"stage": name, "stage_role": plan["stages"][name]["role"],
            "runner": plan["stages"][name]["configuration"].get("runner"),
            "plan_sha256": plan["plan_sha256"], "parameters": len(catalog),
            "steps": plan["steps"], "parameter_steps": len(rows),
            "rank_receipt_sha256": stage.rank_hashes, "raw_files": stage.files,
            "capture_seals_verified": True, "invariant_failures": invariant_failures,
            "delta_checks": len(rows), "continuity_checks": len(catalog) * (len(plan["steps"]) - 1),
            "formula_summary": summarize(rows), "rows": rows}


def audit_suite(path, *, expected_parameters=None, expected_steps=None, verify_live_artifacts=False):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "suite.json"
    suite = read(path)
    suite_hash = suite.pop("suite_sha256", None)
    require(suite_hash == object_hash(suite), "Suite checksum mismatch")
    result = {"schema_version": 1, "status": "complete_report_only", "acceptance_qualified": False,
              "formula_gates_applied": False, "suite_sha256": suite_hash,
              "suite_file_sha256": digest(path), "optimizer": TARGET_OPTIMIZER,
              "formula_conditioning": "Every step uses this side's observed previous parameter/moments and current gradient; step zero starts with zero moments",
              "formulas": {
                  "fp64_explicit": "FP64 mathematical equations; expected parameter rounds to FP32 before capture delta",
                  "fp32_explicit": "Explicit NumPy FP32 non-fused multiply/add/sqrt/divide; not a particular GPU kernel's instruction order"},
              "source_sha256": {"audit": digest(Path(__file__)), "stage_reader": digest(inspect.getsourcefile(Stage))},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "live_input_artifacts_reverified": verify_live_artifacts,
              "limitations": ["No formula residual threshold or numerical qualification is inferred",
                  "Raw finite-precision equations can differ from fused FMA, lerp, rsqrt, and parameter storage rounding",
                  "This audit trusts captured gradients and cannot independently validate forward/backward, masks, or loss normalization",
                  "Canonical owned pieces do not independently prove the actual optimizer group assignment",
                  "Model BF16 copy-back and initial optimizer moments are not captured; fresh zero moments follow the declared contract",
                  "Original live checkpoint/source files are not rehashed unless explicitly requested"],
              "cases": []}
    for case in suite["cases"]:
        plan_path = inside(path.parent, case["plan"])
        plan = load_plan(plan_path)
        require(plan["plan_sha256"] == case["plan_sha256"], "Case plan changed")
        if verify_live_artifacts:
            verify_artifacts(plan)
        stages = [audit_stage(plan_path, name, expected_parameters, expected_steps)
                  for name in plan["stages"]]
        result["cases"].append({"case": case["id"], "stages": stages})
    require(result["cases"] and all(c["stages"] for c in result["cases"]), "No captured cases/stages")
    failures = sum(len(s["invariant_failures"]) for c in result["cases"] for s in c["stages"])
    result["exact_invariant_failures"] = failures
    if failures:
        result["status"] = "invariant_failure"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Existing suite.json or its directory")
    parser.add_argument("--output", required=True, help="New audit JSON; never overwrites")
    parser.add_argument("--expected-parameters", type=int, help="Optional expected canonical parameter count")
    parser.add_argument("--expected-steps", type=int, help="Optional expected fresh step count")
    parser.add_argument("--verify-live-artifacts", action="store_true")
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Audit output already exists")
    try:
        report = audit_suite(args.suite, expected_parameters=args.expected_parameters,
                             expected_steps=args.expected_steps,
                             verify_live_artifacts=args.verify_live_artifacts)
        code = 2 if report["status"] == "invariant_failure" else 0
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        report = {"schema_version": 1, "status": "evidence_failure", "acceptance_qualified": False,
                  "reason": str(error), "source_sha256": digest(Path(__file__))}
        code = 3
    write_new(args.output, report)
    print(json.dumps({"status": report["status"], "acceptance_qualified": False,
                      "output": str(Path(args.output).resolve())}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
