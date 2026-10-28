"""A real Adam update is not sufficient evidence of a valid optimizer counter."""
from types import SimpleNamespace

import pytest
import torch

from slime_parity.torch_capture import TorchStep
from training_parity.common import EvidenceError


@pytest.mark.parametrize("bad", [True, 1.5, float("nan"), float("inf"), "1", [1],
                                 torch.tensor([1.0]), torch.tensor(True), torch.tensor(1.5)])
def test_reference_rejects_invalid_counter_after_real_update(bad):
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    optimizer = torch.optim.AdamW([p], lr=.01)
    ctx = SimpleNamespace(suite={"description": {"parameters": {"w": {"shape": [2]}}}},
                          write=lambda *args: None)
    observer = TorchStep(ctx, 0, {"w": p}, optimizer)
    p.square().sum().backward()
    observer.capture_gradients()
    optimizer.step()
    optimizer.state[p]["step"] = bad
    with pytest.raises(EvidenceError, match="counter"):
        observer.capture_update()


def test_reference_requires_counter_after_real_update():
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    optimizer = torch.optim.AdamW([p], lr=.01)
    ctx = SimpleNamespace(suite={"description": {"parameters": {"w": {"shape": [2]}}}},
                          write=lambda *args: None)
    observer = TorchStep(ctx, 0, {"w": p}, optimizer)
    p.square().sum().backward()
    observer.capture_gradients()
    optimizer.step()
    del optimizer.state[p]["step"]
    with pytest.raises(EvidenceError, match="counter"):
        observer.capture_update()
