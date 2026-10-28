"""Validate the actual Adam counter without coercing malformed evidence."""
import math
from numbers import Real

from training_parity.common import require


def check_counter(value, expected, label):
    import torch

    if isinstance(value, torch.Tensor):
        require(value.ndim == 0 and value.dtype != torch.bool,
                f"{label} counter must be a non-boolean scalar")
        value = value.detach().item()
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) and value == expected,
            f"{label} counter must equal the completed update count {expected}")


def check_native_counter(state, group, expected):
    # PyTorch Adam stores a per-parameter step; TE FusedAdam uses a group step.
    # Accept either layout, but never missing or contradictory evidence.
    require("step" in state or "step" in group, "Missing native Adam counter")
    for label, values in (("Native Adam", state), ("Native Adam group", group)):
        if "step" in values:
            check_counter(values["step"], expected, label)
