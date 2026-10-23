"""Freeze named tensors, steps, stages, identities, and optional acceptance rules."""
from copy import deepcopy
from pathlib import Path
import fnmatch
import math
import re

from . import __version__
from .common import require, read, write_new, object_hash, digest, shape_size

LIMITS = {"max_abs", "relative_l2", "rmse", "cosine_min", "atol", "rtol"}


def selected(rule, name, tensor):
    selectors = rule.get("select", {})
    values = {"names": name, "roles": tensor["role"], "groups": tensor["group"]}
    return all(any(fnmatch.fnmatchcase(values[key], pattern) for pattern in patterns)
               for key, patterns in selectors.items())


def validate(spec):
    require(isinstance(spec, dict) and spec.get("schema_version") == 1, "Unsupported spec schema")
    require(isinstance(spec.get("name"), str) and spec["name"], "Name is required")
    require(isinstance(spec.get("contract"), dict) and spec["contract"], "Declare a shared training contract")
    for key in ("input", "initial_state", "objective"):
        require(spec["contract"].get(key), f"Contract must identify {key}")
    steps = spec.get("steps", [0])
    require(isinstance(steps, list) and steps and all(type(x) is int and x >= 0 for x in steps)
            and len(set(steps)) == len(steps), "Steps must be unique nonnegative integers")
    spec["steps"] = sorted(steps)
    tensors = spec.get("tensors")
    require(isinstance(tensors, dict) and tensors, "Declare a complete tensor catalog")
    for name, tensor in tensors.items():
        require(isinstance(name, str) and name and isinstance(tensor, dict), "Invalid tensor catalog")
        shape_size(tensor.get("shape"))
        tensor.setdefault("group", name.rsplit("/", 1)[0])
        tensor.setdefault("role", "tensor")
        require(all(isinstance(tensor[k], str) and tensor[k] for k in ("group", "role")), "Invalid role or group")
        require(type(tensor.get("require_nonzero", False)) is bool, "require_nonzero must be boolean")
    stages = spec.get("stages")
    require(isinstance(stages, dict) and stages, "Declare stages")
    for name, stage in stages.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name), "Unsafe stage name")
        require(isinstance(stage, dict) and stage.get("role") in ("reference", "repeat", "candidate"), "Invalid stage role")
        stage.setdefault("world_size", 1)
        stage.setdefault("configuration", {})
        require(type(stage["world_size"]) is int and stage["world_size"] > 0, "Invalid world size")
        require(isinstance(stage["configuration"], dict), "Configuration must be an object")
        env = stage.get("environment", {})
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "Invalid environment")
        if "command" in stage:
            command = stage["command"]
            require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command), "Command must be argv")
            require("{plan}" in command and "{stage}" in command, "Command must accept {plan} and {stage} as arguments")
            for value in command:
                for key in re.findall(r"\{([^{}]+)\}", value):
                    require(key in ("plan", "stage", "world_size", "python", "output"), "Unknown command placeholder")
    references = [s for s in stages.values() if s["role"] == "reference"]
    require(len(references) == 1, "Need exactly one reference")
    for stage in stages.values():
        if stage["role"] == "repeat":
            require({k: v for k, v in stage.items() if k != "role"} ==
                    {k: v for k, v in references[0].items() if k != "role"},
                    "Reference repeats must have the same configuration, command, and rank count")
    rules = spec.get("policy", [])
    require(isinstance(rules, list), "Policy must be a list; [] means report_only")
    ids = set()
    for rule in rules:
        require(isinstance(rule, dict) and isinstance(rule.get("id"), str) and rule["id"] and rule["id"] not in ids, "Rule IDs must be unique")
        ids.add(rule["id"])
        require(set(rule) <= {"id", "scope", "select", "limits", "noise"}, "Unknown policy field")
        require(rule.get("scope") in ("tensor", "group", "global"), "Invalid rule scope")
        selectors = rule.get("select", {})
        require(isinstance(selectors, dict) and set(selectors) <= {"names", "roles", "groups"}, "Invalid selectors")
        require(all(isinstance(v, list) and v and all(isinstance(x, str) and x for x in v) for v in selectors.values()), "Selector patterns must be nonempty lists")
        require(any(selected(rule, n, t) for n, t in tensors.items()), "Rule selects no tensors")
        limits = rule.get("limits", {})
        require(isinstance(limits, dict) and not set(limits) - LIMITS, "Unknown metric limit")
        for key, value in limits.items():
            require(type(value) in (int, float) and math.isfinite(value), "Limits must be finite numbers")
            require(-1 <= value <= 1 if key == "cosine_min" else value >= 0, "Invalid limit value")
        require(("atol" in limits) == ("rtol" in limits), "Specify both atol and rtol for elementwise checks")
        noise = rule.get("noise")
        require(limits or noise, "Rule must enforce a limit")
        if noise is not None:
            require(isinstance(noise, dict) and set(noise) == {"repeat_ceiling", "target_ceiling", "multiplier", "slack"}, "Invalid noise policy")
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in noise.values()), "Invalid noise limit")
            require(any(s["role"] == "repeat" for s in stages.values()), "Noise policies require a reference repeat")
    spec["policy"] = rules
    object_hash(spec)


def inventory(path):
    original = Path(path)
    require(not original.is_symlink(), "Artifact symlinks are unsupported")
    path = original.resolve()
    require(path.exists(), "Missing input artifact")
    root = path if path.is_dir() else path.parent
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    require(not any(p.is_symlink() for p in files), "Artifact symlinks are unsupported")
    entries = {str(p.relative_to(root)): {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in files if p.is_file()}
    require(entries, "Artifact must contain files")
    return {"path": str(path), "files": entries, "sha256": object_hash(entries)}


def prepare(spec, output):
    """Freeze a spec dictionary (or JSON path) into a new experiment directory."""
    spec = deepcopy(spec) if isinstance(spec, dict) else read(spec)
    validate(spec)
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; create a new experiment")
    artifacts = {key: inventory(value) for key, value in spec.pop("artifacts", {}).items()}
    require(all(not output.is_relative_to(Path(a["path"])) for a in artifacts.values()), "Output cannot be inside an input artifact")
    plan = {**spec, "artifacts": artifacts, "library_version": __version__}
    plan["plan_sha256"] = object_hash(plan)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", plan)
    return output / "plan.json"


def load_plan(path):
    plan = read(path)
    require(isinstance(plan, dict), "Plan must be an object")
    expected = plan.pop("plan_sha256", None)
    require(object_hash(plan) == expected, "Plan checksum mismatch")
    checked = deepcopy(plan)
    validate(checked)
    require(checked == plan, "Plan is not normalized")
    plan["plan_sha256"] = expected
    return plan


def verify_artifacts(plan):
    for expected in plan["artifacts"].values():
        require(inventory(expected["path"]) == expected, "Frozen input artifact changed")
