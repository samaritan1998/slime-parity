"""Immutable model suites, exhaustive matrices, isolated runs, and coverage gates."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import math
import re
import sys
import uuid

from training_parity.common import require, read, write_new, object_hash, digest, EvidenceError
from training_parity.plan import prepare as prepare_tensors, load_plan, verify_artifacts, selected
from training_parity.runner import run_stage
from training_parity.storage import Stage
from training_parity.compare import compare
from .model import describe_model, tensor_catalog, symbol_source, PARAMETER_ROLES
from .topology import expand_matrix, Topology, validate_groups


def normalize(spec):
    s = deepcopy(spec)
    require(isinstance(s, dict) and s.get("schema_version") == 1, "Unsupported suite schema")
    require(set(s) <= {"schema_version", "name", "model", "reference", "candidate", "training", "cases", "matrix", "policy", "execution", "artifacts"},
            "Unknown suite field (draft design files are not executable suite specifications)")
    require(isinstance(s.get("name"), str) and s["name"], "Suite name is required")
    for field in ("model", "reference", "candidate", "training", "matrix"):
        require(isinstance(s.get(field), dict), f"Missing {field} object")
    require(set(s["model"]) <= {"adapter", "options"} and "adapter" in s["model"], "Invalid model adapter")
    s["model"].setdefault("options", {})
    require(isinstance(s["model"]["options"], dict), "Model options must be an object")
    for field in ("reference", "candidate"):
        runner = s[field]
        require(set(runner) <= {"runner", "kind", "options", "device"} and "runner" in runner, "Invalid runner declaration")
        runner.setdefault("options", {})
        runner.setdefault("device", "cpu" if field == "reference" else "cuda")
        require(runner["device"] in ("cpu", "cuda"), "Unknown execution device")
        require(isinstance(runner["options"], dict), "Runner options must be an object")
    require(s["reference"].get("kind") == "independent", "Declare an independent model reference")
    require(s["candidate"].get("kind") in ("slime", "torch"), "Candidate kind must be slime or torch")
    require(s["reference"]["runner"] != s["candidate"]["runner"], "Reference and candidate cannot be the same entry")
    training = s["training"]
    require(set(training) <= {"steps", "seed", "global_batch_size", "dtype", "optimizer", "clip_grad"}, "Unknown training field")
    require(type(training.get("steps")) is int and training["steps"] >= 2, "State validation needs at least two optimizer steps")
    require(type(training.get("seed")) is int and training["seed"] >= 0, "Declare a nonnegative seed")
    require(type(training.get("global_batch_size")) is int and training["global_batch_size"] > 0, "Declare global batch size")
    training.setdefault("dtype", "float32")
    require(training["dtype"] in ("float32", "float64", "bfloat16"), "Unsupported training dtype")
    optimizer = training.get("optimizer")
    require(isinstance(optimizer, dict) and optimizer.get("name") in ("adam", "adamw"), "Declare Adam/AdamW semantics")
    require(set(optimizer) <= {"name", "lr", "betas", "eps", "weight_decay"}, "Unknown optimizer field")
    for key, default in (("lr", 1e-3), ("eps", 1e-8), ("weight_decay", 0.0)):
        optimizer.setdefault(key, default)
        require(type(optimizer[key]) in (int, float) and math.isfinite(optimizer[key]) and optimizer[key] >= 0, "Invalid optimizer value")
    require(optimizer["eps"] > 0, "Adam epsilon must be positive")
    optimizer.setdefault("betas", [0.9, 0.999])
    require(isinstance(optimizer["betas"], list) and len(optimizer["betas"]) == 2
            and all(type(v) in (int, float) and 0 <= v < 1 for v in optimizer["betas"]), "Invalid Adam betas")
    training.setdefault("clip_grad", 0.0)
    require(type(training["clip_grad"]) in (int, float) and math.isfinite(training["clip_grad"]) and training["clip_grad"] >= 0, "Invalid clipping norm")
    cases = s.get("cases")
    require(isinstance(cases, list) and cases, "Declare test cases")
    ids = []
    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get("id"), str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case["id"]), "Unsafe case ID")
        ids.append(case["id"])
    require(len(set(ids)) == len(ids), "Duplicate case IDs")
    s.setdefault("policy", [])
    execution = s.setdefault("execution", {})
    require(set(execution) <= {"timeout_seconds", "max_world_size", "launcher", "rank_byte_budget"}, "Unknown execution field")
    execution.setdefault("timeout_seconds", 3600)
    require(type(execution["timeout_seconds"]) in (int, float) and math.isfinite(execution["timeout_seconds"]) and execution["timeout_seconds"] > 0, "Invalid timeout")
    execution.setdefault("max_world_size", 1)
    require(type(execution["max_world_size"]) is int and execution["max_world_size"] > 0, "Invalid resource capacity")
    if "rank_byte_budget" in execution:
        require(type(execution["rank_byte_budget"]) is int and execution["rank_byte_budget"] > 0, "Invalid rank byte budget")
    object_hash(s)
    return s


def prepare(spec, output):
    base = Path.cwd() if isinstance(spec, dict) else Path(spec).resolve().parent
    spec = normalize(spec if isinstance(spec, dict) else read(spec))
    # Built-in adapters use local paths. Resolve them before worker cwd changes;
    # custom adapter options remain owned by their adapter contract.
    path_fields = [(spec["candidate"]["options"], ("slime_source", "megatron_source"))]
    if spec["model"]["adapter"] == "slime_parity.adapters.hf:HFCausalLMAdapter":
        path_fields.append((spec["model"]["options"], ("checkpoint", "parameter_map", "native_names")))
        path_fields.extend((case, ("inputs_file",)) for case in spec["cases"])
    for container, keys in path_fields:
        for key in keys:
            if key in container:
                container[key] = str((base / container[key]).resolve())
    if "artifacts" in spec:
        spec["artifacts"] = {name: str((base / value).resolve()) for name,value in spec["artifacts"].items()}
    output = Path(output).resolve()
    require(not output.exists(), "Use a new suite output directory")
    sources = {key: symbol_source(spec[key]["adapter" if key == "model" else "runner"])
               for key in ("model", "reference", "candidate")}
    require(sources["reference"]["definition_sha256"] != sources["candidate"]["definition_sha256"], "Reference implementation is identical to candidate")
    descriptions = [(case, describe_model(spec, case)) for case in spec["cases"]]
    cases = []
    for case, description in descriptions:
        entries = expand_matrix(spec["matrix"], description["constraints"], spec["training"]["global_batch_size"])
        catalog = tensor_catalog(description, PARAMETER_ROLES)
        stages = {}
        for name, kind, topology in [("reference", "reference", None)] + [
                (e["id"], "candidate", e["requested"]) for e in entries if e["eligibility"] == "eligible"]:
            runner = spec[kind]
            world = 1 if kind == "reference" else topology["world_size"]
            command = ["{python}", "-m", "slime_parity.worker", "--plan", "{plan}", "--stage", "{stage}"]
            if "launcher" in spec["execution"]:
                command = spec["execution"]["launcher"]
            elif kind == "candidate" and (world > 1 or runner["kind"] == "slime"):
                command = ["{python}", "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node", "{world_size}", "-m", "slime_parity.worker", "--plan", "{plan}", "--stage", "{stage}"]
            stages[name] = {"role": kind, "world_size": world, "command": command,
                            "configuration": {"runner": runner["runner"], "implementation": runner["kind"],
                                              "device": runner["device"], "options": runner["options"], "topology": topology}}
        artifacts = dict(spec.get("artifacts", {}))
        for key, path in description["artifacts"].items():
            require(key not in artifacts, "Duplicate artifact names")
            artifacts[key] = path
        for prefix, root in (("library_slime_parity", Path(__file__).parent),
                             ("library_comparison", Path(sys.modules["training_parity"].__file__).parent)):
            artifacts.update({prefix + "/" + str(file.relative_to(root)): str(file)
                              for file in sorted(root.rglob("*.py"))})
        for key in ("slime_source", "megatron_source"):
            source = spec["candidate"]["options"].get(key)
            if source:
                root = Path(source).resolve()
                require(root.is_dir(), f"Missing {key} directory")
                # Freeze executable source, excluding caches and repository history.
                artifacts.update({key + "/" + str(file.relative_to(root)): str(file)
                                  for file in sorted(root.rglob("*.py")) if not any(
                                      part in (".git", ".venv", "__pycache__") for part in file.relative_to(root).parts)})
        artifacts.update({f"entry_{key}": value["path"] for key, value in sources.items()})
        low = {"schema_version": 1, "name": f"{spec['name']}/{case['id']}",
               "contract": description["identity"], "steps": list(range(spec["training"]["steps"])),
               "tensors": catalog, "stages": stages, "policy": spec["policy"], "artifacts": artifacts,
               "suite": {"model": spec["model"], "training": spec["training"], "case": case,
                         "description": description, "sources": sources, "execution": spec["execution"],
                         "input_identities": {str(i): object_hash(x) for i,x in enumerate(description["inputs"])}}}
        plan_path = prepare_tensors(low, output / "cases" / case["id"])
        cases.append({"id": case["id"], "plan": str(plan_path.relative_to(output)),
                      "plan_sha256": load_plan(plan_path)["plan_sha256"], "matrix": entries})
    suite = {"schema_version": 1, "spec": spec, "sources": sources, "cases": cases}
    suite["suite_sha256"] = object_hash(suite)
    write_new(output / "suite.json", suite)
    return output / "suite.json"


def load_suite(path):
    path = Path(path).resolve()
    suite = read(path)
    expected = suite.pop("suite_sha256", None)
    require(expected == object_hash(suite), "Suite checksum mismatch")
    require(normalize(suite["spec"]) == suite["spec"], "Suite spec is not normalized")
    for case in suite["cases"]:
        from training_parity.common import inside
        plan = load_plan(inside(path.parent, case["plan"]))
        require(plan["plan_sha256"] == case["plan_sha256"], "Case plan changed")
    suite["suite_sha256"] = expected
    return suite


def check_capture(stage):
    plan = stage.plan
    names = [r["stage"] for r in stage.ranks]
    require(len(set(names)) == 1, "Mixed stage evidence")
    configuration = plan["stages"][names[0]]["configuration"]
    for row in stage.ranks:
        meta = row["metadata"]
        require(meta.get("adapter") == configuration["runner"] and meta.get("implementation") == configuration["implementation"], "Capture implementation differs")
        steps = meta.get("steps", [])
        require([r["step"] for r in steps] == plan["steps"], "Missing or out-of-order observed steps")
        for step in steps:
            require(step.get("optimizer_executed") is True and
                    step.get("input_identity") == plan["suite"]["input_identities"][str(step["step"])], "Incomplete native step evidence")
    if configuration["topology"]:
        validate_groups(configuration["topology"], [r["metadata"]["parallelism"] for r in stage.ranks])


def stage_status(path, name):
    folder = path.parent / name
    if (folder / "complete.json").is_file():
        try:
            stage = Stage(path, name)
            check_capture(stage)
            return {"execution": "complete", "validation": "unverified"}
        except (EvidenceError, KeyError, OSError, TypeError, ValueError) as error:
            return {"execution": "complete", "validation": "incomplete_evidence", "reason": str(error)}
    if (folder / "failure.json").is_file():
        failure = read(folder / "failure.json")
        return {"execution": "execution_failure", "validation": "unverified", "reason": failure.get("error", "Stage failed")}
    if (folder / "unavailable.json").is_file():
        return {"execution": "resource_unavailable", "validation": "unverified", "reason": read(folder / "unavailable.json")["reason"]}
    return {"execution": "incomplete" if folder.exists() else "planned", "validation": "unverified"}


def report(path):
    path = Path(path).resolve()
    suite = load_suite(path)
    rows, references = [], []
    for case in suite["cases"]:
        plan_path = path.parent / case["plan"]
        plan = load_plan(plan_path)
        reference = stage_status(plan_path, "reference")
        references.append({"case": case["id"], **reference})
        artifacts_valid = True
        try:
            verify_artifacts(plan)
        except (EvidenceError, OSError) as error:
            artifacts_valid = False
            reference = {"execution": reference["execution"], "validation": "incomplete_evidence", "reason": str(error)}
            references[-1].update(reference)
        for entry in case["matrix"]:
            row = {"case": case["id"], **entry}
            if entry["eligibility"] != "eligible":
                row.update(execution=entry["eligibility"], validation="unverified")
            else:
                row.update(stage_status(plan_path, entry["id"]))
                if row["execution"] == "complete" and row["validation"] != "incomplete_evidence":
                    if artifacts_valid and reference["execution"] == "complete" and reference["validation"] != "incomplete_evidence":
                        try:
                            comparison = compare(plan_path, candidates=[entry["id"]])
                            row["comparison"] = comparison
                            row["validation"] = comparison["status"]
                            # Every declared tensor needs an individual numerical gate.
                            missing = [n for n, tensor in plan["tensors"].items() if not any(
                                r["scope"] == "tensor" and selected(r,n,tensor) for r in plan["policy"])]
                            row["ungated_required_tensors"] = missing
                            if missing and row["validation"] == "passed":
                                row.update(validation="report_only", reason="Required tensors lack individual acceptance rules")
                        except (EvidenceError, OSError, ValueError, KeyError) as error:
                            row.update(validation="incomplete_evidence", reason=str(error))
                    else:
                        row.update(validation="unverified", reason="Independent reference or frozen artifacts unavailable")
            rows.append(row)
    required = [r for r in rows if r["eligibility"] == "eligible"]
    counts = {"requested": len(rows), "required": len(required),
              "completed": sum(r["execution"] == "complete" for r in required),
              "passed": sum(r["validation"] == "passed" for r in required),
              "validation": dict(Counter(r["validation"] for r in required)),
              "execution": dict(Counter(r["execution"] for r in rows))}
    if required and all(r["validation"] == "passed" for r in required):
        status = "passed"
    elif any(r["validation"] == "numerical_failure" for r in required):
        status = "numerical_failure"
    elif required and all(r["execution"] == "complete" and r["validation"] in ("passed", "report_only") for r in required):
        status = "report_only"
    else:
        status = "unverified"
    return {"schema_version": 1, "name": suite["spec"]["name"], "status": status,
            "suite_sha256": suite["suite_sha256"], "counts": counts, "references": references,
            "rows": rows, "reference_provenance": suite["sources"]["reference"],
            "scope": {"model": suite["spec"]["model"], "training": suite["spec"]["training"],
                      "matrix": suite["spec"]["matrix"], "candidate_kind": suite["spec"]["candidate"]["kind"]},
            "limitations": ["Pass applies to declared cases, steps, tensors, policies and legal requested configurations",
                            "Independent reference semantics and model adapter declarations require review",
                            "CPU fixture results do not establish Slime GPU support"]}


def execute(path):
    path = Path(path).resolve()
    suite = load_suite(path)
    for case in suite["cases"]:
        plan_path = path.parent / case["plan"]
        plan = load_plan(plan_path)
        if not any(entry["eligibility"] == "eligible" for entry in case["matrix"]):
            continue  # Preserve the full rejected domain without launching a reference.
        for name, stage in plan["stages"].items():
            folder = plan_path.parent / name
            if folder.exists():
                continue  # Never overwrite a completed, failed, or interrupted attempt.
            reason = None
            if stage["world_size"] > suite["spec"]["execution"]["max_world_size"]:
                reason = "Requested world size exceeds declared resource capacity"
            elif stage["configuration"]["device"] == "cuda" and "launcher" not in suite["spec"]["execution"]:
                try:
                    import torch
                    if not torch.cuda.is_available() or ("launcher" not in suite["spec"]["execution"] and torch.cuda.device_count() < stage["world_size"]):
                        reason = "Insufficient locally visible CUDA devices"
                except ImportError:
                    reason = "PyTorch CUDA environment is unavailable"
            if reason:
                write_new(folder / "unavailable.json", {"reason": reason})
                continue
            try:
                run_stage(plan_path, name, timeout=suite["spec"]["execution"]["timeout_seconds"])
            except Exception as error:
                # run_stage retains exact failure and log. Other configurations
                # remain independent; one failure must not discard their evidence.
                if not (folder / "failure.json").exists():
                    write_new(folder / "failure.json", {"status": "execution_failure", "error": str(error)})
    result = report(path)
    from .reporting import save_report
    destination = path.parent / "reports" / uuid.uuid4().hex
    save_report(result, destination)
    result["report_directory"] = str(destination)
    return result
