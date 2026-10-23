"""Compare all declared steps; evaluate only frozen, explicitly selected rules."""
import math
from pathlib import Path
import numpy as np

from .common import require, object_hash
from .metrics import Accumulator
from .plan import load_plan, selected
from .storage import Stage, aligned_chunks


def unit_for(rule, name, tensor):
    return name if rule["scope"] == "tensor" else tensor["group"] if rule["scope"] == "group" else "*"


def pair_metrics(reference, actual, plan):
    output, rule_metrics = {}, {}
    for step in plan["steps"]:
        tensors, groups, roles = {}, {}, {}
        for name, tensor in plan["tensors"].items():
            accumulator = Accumulator()
            cursor, largest, diagnostic = 0, -1.0, None
            active = [rule for rule in plan["policy"] if selected(rule, name, tensor)]
            for left, right in aligned_chunks(reference, actual, (step, name)):
                part = Accumulator().add(left, right)
                accumulator.merge(part)
                difference = np.abs(right.astype(np.float64) - left.astype(np.float64))
                index = int(np.argmax(difference))
                if float(difference[index]) > largest:
                    largest = float(difference[index])
                    diagnostic = {"flat_index": cursor + index, "reference": float(left[index]),
                                  "candidate": float(right[index])}
                cursor += left.size
                for rule in active:
                    key = (rule["id"], step, unit_for(rule, name, tensor))
                    item = rule_metrics.setdefault(key, {"stats": Accumulator(), "violations": 0})
                    item["stats"].merge(part)
                    if "atol" in rule.get("limits", {}):
                        limits = rule["limits"]
                        x, y = left.astype(np.float64), right.astype(np.float64)
                        with np.errstate(over="ignore", invalid="ignore"):
                            allowance = limits["atol"] + limits["rtol"] * np.abs(x)
                        require(np.isfinite(allowance).all(), "Elementwise tolerance overflow")
                        item["violations"] += int(np.count_nonzero(np.abs(y - x) > allowance))
            row = accumulator.result()
            diagnostic["index"] = [int(i) for i in np.unravel_index(diagnostic["flat_index"], tensor["shape"])]
            row["max_abs_location"] = diagnostic
            if tensor.get("require_nonzero", False):
                require(row["reference_l2"] > 0 and row["candidate_l2"] > 0, f"Required nonzero tensor is zero: {name}")
            tensors[name] = row
            groups.setdefault(tensor["group"], Accumulator()).merge(accumulator)
            roles.setdefault(tensor["role"], Accumulator()).merge(accumulator)
        output[str(step)] = {"tensors": tensors,
                             "groups": {n: a.result() for n, a in groups.items()},
                             "roles": {n: a.result() for n, a in roles.items()}}
    return output, {k: {"metrics": v["stats"].result(), "elementwise_violations": v["violations"]}
                    for k, v in rule_metrics.items()}


def fixed_check(measured, limits):
    values, reasons = measured["metrics"], []
    for metric in ("max_abs", "relative_l2", "rmse"):
        if metric in limits and (values[metric] is None or values[metric] > limits[metric]):
            reasons.append(f"{metric} exceeds the frozen limit or is undefined")
    if "cosine_min" in limits and values["cosine"] < limits["cosine_min"]:
        reasons.append("cosine is below the frozen minimum")
    if "atol" in limits and measured["elementwise_violations"]:
        reasons.append("elementwise absolute-plus-relative tolerance exceeded")
    return reasons


def compare(plan_path, *, candidates=None, reference=None):
    """Read all sealed evidence and return a JSON-serializable report.

    No files are deleted or rewritten. A report_only result is never promoted to
    passed by zero differences. Failed/incomplete evidence raises EvidenceError.
    """
    plan = load_plan(plan_path)
    original_reference = next(n for n, s in plan["stages"].items() if s["role"] == "reference")
    reference = original_reference if reference is None else reference
    require(reference in plan["stages"], "Unknown reference stage")
    if reference != original_reference:
        require(not any(r.get("noise") for r in plan["policy"]), "Alternative reference needs its own frozen noise calibration")
    repeats = [n for n, s in plan["stages"].items() if s["role"] == "repeat"] if reference == original_reference else []
    candidates = ([n for n, s in plan["stages"].items() if s["role"] == "candidate" and n != reference]
                  if candidates is None else list(candidates))
    require(candidates and len(set(candidates)) == len(candidates)
            and all(n in plan["stages"] and plan["stages"][n]["role"] == "candidate" and n != reference for n in candidates),
            "Select distinct candidate stages different from reference")
    stages = {name: Stage(plan_path, name) for name in [reference, *repeats, *candidates]}
    comparisons, measured = {}, {}
    for name in repeats + candidates:
        comparisons[name], measured[name] = pair_metrics(stages[reference], stages[name], plan)
    gates = []
    for rule in plan["policy"]:
        keys = [k for k in measured[candidates[0]] if k[0] == rule["id"]]
        for key in keys:
            noise = rule.get("noise")
            target_limit = None
            noise_invalid = False
            if noise:
                observed = [measured[n][key]["metrics"]["relative_l2"] for n in repeats]
                noise_invalid = any(x is None or x > noise["repeat_ceiling"] for x in observed)
                if not any(x is None for x in observed):
                    worst = max(observed)
                    target_limit = min(noise["target_ceiling"], max(noise["multiplier"] * worst, worst + noise["slack"]))
                    require(math.isfinite(target_limit), "Noise threshold overflow")
            checks = {}
            for name in repeats + candidates:
                item = measured[name][key]
                reasons = fixed_check(item, rule.get("limits", {}))
                if noise:
                    delta = item["metrics"]["relative_l2"]
                    if name in repeats:
                        if delta is None or delta > noise["repeat_ceiling"]:
                            reasons.append("reference repeat exceeds ceiling or has undefined relative error")
                    elif noise_invalid or target_limit is None or delta is None or delta > target_limit:
                        reasons.append("candidate exceeds noise-aware limit, or the repeat is invalid")
                checks[name] = {"passed": not reasons, "reasons": reasons, **item}
            gates.append({"rule": rule["id"], "scope": rule["scope"], "step": key[1], "unit": key[2],
                          "noise_relative_limit": target_limit, "checks": checks,
                          "passed": all(c["passed"] for c in checks.values())})
    status = "report_only" if not gates else "passed" if all(g["passed"] for g in gates) else "numerical_failure"
    coverage = {}
    for name, tensor in plan["tensors"].items():
        coverage[name] = [{"id": rule["id"], "scope": rule["scope"]}
                          for rule in plan["policy"] if selected(rule, name, tensor)]
    return {"schema_version": 1, "library_version": plan["library_version"],
            "name": plan["name"], "status": status, "plan_sha256": plan["plan_sha256"],
            "contract": plan["contract"], "steps": plan["steps"], "reference_stage": reference,
            "policy": plan["policy"], "comparisons": comparisons, "gates": gates,
            "coverage": coverage, "ungated_tensors": [n for n, rules in coverage.items() if not rules],
            "stages": {n: {"configuration": plan["stages"][n]["configuration"],
                           "process": s.process, "rank_receipts": s.rank_hashes,
                           "raw_artifacts": s.files,
                           "rank_metadata": [r.get("metadata", {}) for r in s.ranks]}
                       for n, s in stages.items()},
            "raw_retained": True,
            "limits": ["Passed applies only to the listed frozen rules and observed steps",
                       "The adapter is trusted to report actual training tensors and identities",
                       "Matching checksums prove internal consistency, not semantic correctness",
                       "Metadata and configuration differences are recorded, not hidden by a runtime-equality requirement"]}
