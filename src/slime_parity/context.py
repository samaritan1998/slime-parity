"""Capture context shared by independent references and native adapters."""
from pathlib import Path
import os
import platform
import importlib.metadata

import numpy as np
from training_parity.common import require, object_hash
from training_parity.plan import load_plan, verify_artifacts
from training_parity.storage import RankWriter
from .mapping import torch_chunks


def runtime_environment():
    versions = {}
    for name in ("slime-parity", "numpy", "torch", "transformers", "megatron-core", "transformer-engine"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    try:
        import torch
        result["cuda_build"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        result["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return result


class CaptureContext:
    def __init__(self, plan_path, stage):
        self.path = Path(plan_path).resolve()
        self.plan = load_plan(self.path)
        verify_artifacts(self.plan)
        self.name = stage
        self.stage = self.plan["stages"][stage]
        self.suite = self.plan["suite"]
        self.rank = int(os.environ.get("RANK", "0"))
        require(int(os.environ.get("WORLD_SIZE", "1")) == self.stage["world_size"], "Launcher world size differs")
        self.writer = RankWriter(self.path, stage, self.rank,
                                 max_bytes=self.suite["execution"].get("rank_byte_budget"))
        self.steps, self.metadata = [], {"steps": [], "adapter": self.stage["configuration"]["runner"],
                                        "implementation": self.stage["configuration"]["implementation"],
                                        "environment": runtime_environment()}

    @property
    def topology(self):
        return self.stage["configuration"].get("topology")

    def write(self, step, role, name, value):
        key = f"{role}/{name}"
        if hasattr(value, "detach"):
            self.writer.append(step, key, 0, value.numel(), torch_chunks(value))
        else:
            self.writer.write(step, key, np.asarray(value, dtype=np.float64))

    def observe(self, step, name, value):
        self.write(step, "observation", name, value)

    def record_step(self, step, *, input_identity, optimizer_executed, details=None):
        require(step in self.plan["steps"] and step not in self.steps, "Duplicate or unexpected optimizer step")
        require(optimizer_executed is True, "Native optimizer did not execute")
        expected = self.suite["input_identities"][str(step)]
        require(input_identity == expected, "Observed global training input differs from frozen case")
        self.steps.append(step)
        self.metadata["steps"].append({"step": step, "input_identity": input_identity,
                                       "optimizer_executed": True, "details": details or {}})

    def finish(self):
        require(sorted(self.steps) == self.plan["steps"], "Capture did not observe every requested step")
        return self.writer.finish(observed_contract=self.plan["contract"],
                    configuration=self.stage["configuration"], completed_steps=self.steps, metadata=self.metadata)

    def close(self):
        self.writer.close()


def input_identity(value):
    """Hash canonical JSON inputs in logical sample order, before partitioning."""
    return object_hash(value)
