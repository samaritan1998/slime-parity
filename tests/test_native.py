"""Observer unit tests with real Adam/Gloo; no claim of Megatron GPU execution."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist

from slime_parity.mapping import MegatronMapping, Interval
from slime_parity.native import SlimeStepObserver, master_parameter, check_optimizer_options
from slime_parity.topology import Topology
from training_parity.common import EvidenceError


class MemoryWriter:
    def __init__(self, root):
        self.rank_dir = root
        self.rows = {}

    def append(self, step, name, start, end, chunks):
        assert start == 0 and (step, name) not in self.rows
        array = np.concatenate(list(chunks))
        assert array.size == end
        self.rows[step, name] = array.copy()


class NativeAdamHarness:
    """Implements only the Megatron observer boundary, backed by torch AdamW."""
    def __init__(self, model_parameter):
        self.model_parameter = model_parameter
        self.master = torch.nn.Parameter(model_parameter.detach().flatten().clone())
        self.optimizer = torch.optim.AdamW([self.master], lr=.01, weight_decay=0)
        self.model_param_group_index_map = {model_parameter: (0, 0)}

    def _get_model_param_range_map(self, p):
        return {"param": SimpleNamespace(start=0, end=p.numel())}

    def prepare_grads(self):
        self.master.grad = self.model_parameter.grad.flatten().clone()
        return False

    def step(self):
        self.prepare_grads()
        norm = torch.nn.utils.clip_grad_norm_([self.master], .5)
        self.optimizer.step()
        return True, norm, 0


@pytest.fixture
def observer_fixture(tmp_path, monkeypatch):
    utils = ModuleType("megatron.core.utils")
    utils.unwrap_model = lambda models: models
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    topology = Topology().as_dict()
    monkeypatch.setattr("slime_parity.native.observe_megatron_groups", lambda *_: {"topology": topology})
    p = torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3) / 10)
    mapping = object.__new__(MegatronMapping)
    mapping.parameters = {"local": p}
    mapping.intervals = {"local": [Interval("weight", (2, 3), 0, 6, 0)]}
    expected = {"name": "adamw", "lr": .01, "eps": 1e-8, "weight_decay": 0, "betas": [.9, .999]}
    ctx = SimpleNamespace(suite={"description": {"parameters": {"weight": {"shape": [2, 3]}}},
                                 "training": {"optimizer": expected}}, topology=topology,
                          metadata={}, writer=MemoryWriter(tmp_path))
    try:
        yield ctx, p, mapping, NativeAdamHarness(p)
    finally:
        dist.destroy_process_group()


def test_observe_three_actual_updates_and_preclip_gradients(observer_fixture):
    ctx, p, mapping, native = observer_fixture
    original_prepare, original_step = native.prepare_grads, native.step
    for step in range(3):
        p.grad = torch.arange(6, dtype=torch.float64).reshape(2, 3) + 10 + step
        before = native.master.detach().numpy().copy()
        with SlimeStepObserver(ctx, step, [object()], native, None, mapping=mapping):
            native.step()
        rows = ctx.writer.rows
        np.testing.assert_array_equal(rows[step, "gradient/weight"], p.grad.flatten().numpy())
        np.testing.assert_array_equal(rows[step, "initial/weight"], before)
        np.testing.assert_array_equal(rows[step, "update/weight"], native.master.detach().numpy() - before)
        np.testing.assert_array_equal(rows[step, "parameter/weight"], native.master.detach().numpy())
        for field in ("exp_avg", "exp_avg_sq"):
            np.testing.assert_array_equal(rows[step, field+"/weight"], native.optimizer.state[native.master][field].numpy())
        assert native.master.grad.norm() <= .500001
    assert native.prepare_grads == original_prepare and native.step == original_step
    assert not list(ctx.writer.rank_dir.glob("before-step-*"))


def test_missing_update_is_not_a_successful_step(observer_fixture):
    ctx, p, mapping, native = observer_fixture
    original = native.step
    with pytest.raises(EvidenceError, match="not fully observed"):
        with SlimeStepObserver(ctx, 0, [object()], native, None, mapping=mapping):
            pass
    assert native.step == original
    assert not list(ctx.writer.rank_dir.glob("before-step-*"))


def test_skipped_native_optimizer_is_rejected(observer_fixture):
    ctx, p, mapping, native = observer_fixture
    native.step = lambda: (False, 0, 0)
    with pytest.raises(EvidenceError, match="skipped or failed"):
        with SlimeStepObserver(ctx, 0, [object()], native, None, mapping=mapping):
            native.step()


def test_bfloat16_signed_master_remainders_reconstructed_exactly():
    # Select both positive and negative rounding residuals, including a BF16
    # rounding carry. This is bit reconstruction, not BF16 plus a float delta.
    full = torch.tensor([1.002, 1.006, -1.002, -1.006, .000123], dtype=torch.float32)
    bf16 = torch.nn.Parameter(full.to(torch.bfloat16))
    upper = bf16.detach().view(torch.int16).to(torch.int32) << 16
    residual = (full.view(torch.int32) - upper).to(torch.int16)
    optimizer = SimpleNamespace(state={bf16: {"master_param": residual}}, store_param_remainders=True)
    torch.testing.assert_close(master_parameter(optimizer, bf16), full, rtol=0, atol=0)


def test_low_precision_master_cannot_be_guessed():
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(EvidenceError, match="no master"):
        master_parameter(SimpleNamespace(state={}), p)


def test_actual_optimizer_configuration_must_match(observer_fixture):
    ctx, p, mapping, native = observer_fixture
    native.optimizer.param_groups[0]["lr"] *= 2
    with pytest.raises(EvidenceError, match="lr differs"):
        SlimeStepObserver(ctx, 0, [object()], native, None, mapping=mapping)


@pytest.mark.parametrize("counter", [None, True, 1.5, float("nan"), float("inf"), "1",
                                      torch.tensor([1.0]), torch.tensor(True)])
@pytest.mark.parametrize("location", ["state", "group"])
def test_native_rejects_invalid_counter_after_real_update(observer_fixture, counter, location):
    ctx, p, mapping, native = observer_fixture
    actual_step = native.optimizer.step

    def corrupt_counter():
        result = actual_step()
        state = native.optimizer.state[native.master]
        del state["step"]
        if counter is not None:
            target = state if location == "state" else native.optimizer.param_groups[0]
            target["step"] = counter
        return result

    native.optimizer.step = corrupt_counter
    p.grad = torch.ones_like(p)
    with pytest.raises(EvidenceError, match="counter"):
        with SlimeStepObserver(ctx, 0, [object()], native, None, mapping=mapping):
            native.step()
    assert not list(ctx.writer.rank_dir.glob("before-step-*"))


def test_native_accepts_group_counter_and_checks_conflicts(observer_fixture):
    # TE FusedAdam stores the counter in its group, while torch AdamW stores it
    # per parameter. Both are supported, but conflicting present counters fail.
    ctx, p, mapping, native = observer_fixture
    actual_step = native.optimizer.step

    def group_counter():
        result = actual_step()
        state = native.optimizer.state[native.master]
        native.optimizer.param_groups[0]["step"] = state.pop("step")
        return result

    native.optimizer.step = group_counter
    p.grad = torch.ones_like(p)
    with SlimeStepObserver(ctx, 0, [object()], native, None, mapping=mapping):
        native.step()
    assert len(ctx.writer.rows) == 6

    native.optimizer.state[native.master]["step"] = torch.tensor(1.)

    def conflicting_counter():
        result = actual_step()
        # Parameter counter advanced to 2, group counter remains at 1.
        return result

    native.optimizer.step = conflicting_counter
    with pytest.raises(EvidenceError, match="counter"):
        with SlimeStepObserver(ctx, 1, [object()], native, None, mapping=mapping):
            native.step()
