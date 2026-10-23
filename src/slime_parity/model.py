"""One model adapter supplies semantics; the suite owns topology enumeration."""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
from typing import Protocol

from training_parity.common import digest, object_hash, require, shape_size

PARAMETER_ROLES = ("initial", "gradient", "update", "parameter", "exp_avg", "exp_avg_sq")


class ModelAdapter(Protocol):
    def describe(self, case: dict, training: dict) -> dict:
        """Return parameters, observations, constraints, identity, and artifacts.

        Parameters are canonical trainable tensors; observations include loss
        and forward outputs. Identity describes fixed inputs/weights/objective.
        The catalog is derived independently of candidate rank output.
        """
        ...


def load_symbol(path):
    require(isinstance(path, str) and path.count(":") == 1, "Use module:symbol import paths")
    module_name, symbol = path.split(":")
    require(module_name and symbol and not symbol.startswith("_"), "Invalid public import path")
    module = importlib.import_module(module_name)
    value = module
    for part in symbol.split("."):
        require(part.isidentifier() and not part.startswith("_"), "Invalid symbol component")
        value = getattr(value, part)
    require(callable(value), "Adapter entry must be callable")
    return value


def symbol_source(path):
    value = load_symbol(path)
    source = inspect.getsourcefile(value)
    require(source and Path(source).is_file(), "Adapter must expose a source file for freezing")
    return {"symbol": path, "path": str(Path(source).resolve()), "sha256": digest(source),
            "definition_sha256": object_hash(inspect.getsource(value))}


def describe_model(spec, case):
    adapter = load_symbol(spec["model"]["adapter"])(deepcopy(spec["model"].get("options", {})))
    value = adapter.describe(deepcopy(case), deepcopy(spec["training"]))
    require(isinstance(value, dict) and set(value) <= {"parameters", "observations", "constraints", "identity", "artifacts", "inputs"},
            "Invalid model description fields")
    parameters = value.get("parameters")
    require(isinstance(parameters, dict) and parameters, "Model must declare canonical trainable parameters")
    for name, tensor in parameters.items():
        require(isinstance(name, str) and name and isinstance(tensor, dict), "Invalid canonical parameter")
        shape_size(tensor.get("shape"))
        require(set(tensor) <= {"shape", "group"}, "Unknown parameter field")
        tensor.setdefault("group", name.rsplit(".", 1)[0])
    observations = value.get("observations")
    require(isinstance(observations, dict) and "loss" in observations, "Declare loss and forward observations")
    require(any(o.get("role") == "forward" for o in observations.values()), "A full suite requires forward observations")
    for name, tensor in observations.items():
        require(isinstance(name, str) and name and isinstance(tensor, dict), "Invalid observation")
        shape_size(tensor.get("shape"))
        require(set(tensor) <= {"shape", "group", "role"}, "Unknown observation field")
        require(isinstance(tensor.get("role"), str) and tensor["role"], "Observation role is required")
        tensor.setdefault("group", "observations")
    require(observations["loss"]["role"] == "loss", "Loss observation must have the loss role")
    identity = value.get("identity", {})
    require(set(identity) == {"input", "initial_state", "objective"} and all(identity.values()), "Model identity is incomplete")
    require(isinstance(value.get("inputs"), list) and len(value["inputs"]) == spec["training"]["steps"],
            "Provide fixed inputs for every optimizer step")
    require(identity["input"] == object_hash(value["inputs"]), "Input identity differs from actual fixed inputs")
    constraints = value.setdefault("constraints", {})
    require(set(constraints) <= {"num_experts", "num_layers", "hidden_size", "num_attention_heads", "ffn_hidden_size", "tp_divisible", "uniform_pipeline"},
            "Unknown model constraint")
    for key in ("num_experts", "num_layers", "hidden_size", "num_attention_heads", "ffn_hidden_size"):
        if key in constraints:
            require(type(constraints[key]) is int and constraints[key] >= (0 if key == "num_experts" else 1), "Invalid model dimension")
    require(type(constraints.get("uniform_pipeline", False)) is bool, "uniform_pipeline must be boolean")
    require(isinstance(constraints.get("tp_divisible", []), list)
            and set(constraints.get("tp_divisible", [])) <= {"hidden_size", "num_attention_heads", "ffn_hidden_size"}
            and all(k in constraints for k in constraints.get("tp_divisible", [])), "Invalid divisibility constraints")
    value.setdefault("artifacts", {})
    require(isinstance(value["artifacts"], dict), "Artifacts must be a mapping of names to paths")
    return value


def tensor_catalog(description, roles):
    result = {}
    for name, param in description["parameters"].items():
        for role in roles:
            result[f"{role}/{name}"] = {"shape": param["shape"], "group": param["group"], "role": role}
    for name, observation in description["observations"].items():
        result[f"observation/{name}"] = dict(observation)
    return result
