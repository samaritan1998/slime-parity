"""Replay checks use real torch optimizer capture and targeted independent faults."""
from copy import deepcopy
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

import importlib.util
import sys
from pathlib import Path

_DIAGNOSTICS = Path(__file__).resolve().parents[1] / "examples" / "diagnostics" / "bf16"


def _load_diagnostic(name):
    path = _DIAGNOSTICS / (name + ".py")
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

audit = _load_diagnostic("adamw_replay_audit")
from training_parity.common import EvidenceError
from training_parity.plan import prepare, load_plan
from training_parity.storage import RankWriter, finalize_stage


def captured_sequence():
    # Values include epsilon-scale gradients and zero, but useful representable
    # updates. This synthetic check does not choose thresholds for GPU captures.
    p = torch.nn.Parameter(torch.tensor([0., .01, -.01, 1., -1., 1e-7]))
    optimizer = torch.optim.AdamW([p], lr=1e-5, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0, foreach=False, fused=False)
    gradients = [[0., 1e-10, -1e-8, .01, -.2, 1e-6],
                 [1e-9, -2e-10, 1e-8, -.02, .1, -3e-6],
                 [-1e-9, 0., -1e-8, .03, .2, 2e-6]]
    captures = []
    for gradient in gradients:
        before = p.detach().clone()
        p.grad = torch.tensor(gradient)
        optimizer.step()
        state = optimizer.state[p]
        values = {"initial": before, "gradient": p.grad, "parameter": p.detach(),
                  "update": p.detach() - before, "exp_avg": state["exp_avg"],
                  "exp_avg_sq": state["exp_avg_sq"]}
        captures.append({k: v.detach().numpy().copy() for k, v in values.items()})
    return captures


def test_fp64_formula_against_decimal_and_epsilon_behavior():
    g = np.array([0., 1e-10, -1e-8, 1e-6], dtype=np.float32)
    z = np.zeros_like(g)
    expected = audit.formula_step(z, g, z, z, 1, np.float64)
    with localcontext() as context:
        context.prec = 60
        exact = [-Decimal('0.00001') * Decimal(float(x)) /
                 (abs(Decimal(float(x))) + Decimal('0.00000001')) for x in g]
    np.testing.assert_allclose(expected["mathematical_update"], list(map(float, exact)), rtol=3e-15, atol=0)
    assert abs(expected["mathematical_update"][1]) < 1e-5 / 50
    assert abs(expected["mathematical_update"][3]) > .98e-5


def test_torch_trajectory_has_complete_invariants_and_small_formula_residuals():
    rows = audit.audit_tensor(captured_sequence())
    assert len(rows) == 3
    assert all(check["passed"] for row in rows for check in row["invariants"].values())
    assert rows[0]["diagnostics"]["gradient_zero"] == 1
    assert rows[0]["diagnostics"]["gradient_nonzero_abs_le_epsilon"] > 0
    for row in rows:
        for mode in audit.FORMULAS:
            for role in ("exp_avg", "exp_avg_sq"):
                # Test-fixture-only rounding bound; the production audit never
                # uses this number as an acceptance tolerance.
                assert row["formulas"][mode][role]["max_abs"] < 1e-8
            assert row["formulas"][mode]["update"]["max_abs"] < 2e-9


@pytest.mark.parametrize("fault", ["skip_update", "flip_update", "reset_moment", "wrong_gradient", "wrong_step"])
def test_targeted_fault_has_large_own_state_replay_residual(fault):
    captures = captured_sequence()
    clean = audit.audit_tensor(captures)
    bad = deepcopy(captures)
    step, coordinate = 1, 3
    if fault == "skip_update":
        bad[step]["parameter"] = bad[step]["initial"].copy()
        bad[step]["update"][:] = 0
        role = "update"
    elif fault == "flip_update":
        bad[step]["parameter"][coordinate] = bad[step]["initial"][coordinate] - bad[step]["update"][coordinate]
        bad[step]["update"] = bad[step]["parameter"] - bad[step]["initial"]
        role = "update"
    elif fault == "reset_moment":
        bad[step]["exp_avg"][:] = 0
        role = "exp_avg"
    elif fault == "wrong_gradient":
        bad[step]["gradient"][coordinate] *= -1
        role = "exp_avg"
    else:
        # Wrong bias-correction counter; keep m/v and captured delta internally
        # self-consistent so only the independent recurrence reveals the error.
        prediction = audit.formula_step(bad[step]["initial"], bad[step]["gradient"],
                                        bad[step-1]["exp_avg"], bad[step-1]["exp_avg_sq"], 1, np.float64)
        bad[step]["parameter"] = prediction["parameter"]
        bad[step]["update"] = bad[step]["parameter"] - bad[step]["initial"]
        role = "update"
    result = audit.audit_tensor(bad)
    healthy_error = clean[step]["formulas"]["fp64_explicit"][role]["max_abs"]
    fault_error = result[step]["formulas"]["fp64_explicit"][role]["max_abs"]
    assert fault_error > max(100 * healthy_error, 1e-7)
    assert result[step]["invariants"]["captured_delta"]["passed"]


def test_continuity_and_delta_detect_coherent_and_incoherent_records():
    bad = captured_sequence()
    bad[1]["initial"][0] += .01
    bad[1]["update"] = bad[1]["parameter"] - bad[1]["initial"]
    result = audit.audit_tensor(bad)
    assert not result[1]["invariants"]["master_continuity"]["passed"]
    assert result[1]["invariants"]["captured_delta"]["passed"]
    bad[2]["update"][0] = 1
    assert not audit.audit_tensor(bad)[2]["invariants"]["captured_delta"]["passed"]


def test_capture_delta_is_distinct_from_mathematical_update():
    p = np.array([1e10], dtype=np.float32)
    g = np.array([.1], dtype=np.float32)
    z = np.zeros_like(g)
    result = audit.formula_step(p, g, z, z, 1, np.float64)
    assert result["mathematical_update"][0] != 0
    assert result["update"][0] == 0
    assert result["parameter"][0] == p[0]


@pytest.mark.parametrize("bad", ["nan", "negative_v", "missing_role", "float64"])
def test_reject_incomplete_or_invalid_state(bad):
    values = captured_sequence()
    if bad == "nan": values[1]["gradient"][0] = np.nan
    elif bad == "negative_v": values[1]["exp_avg_sq"][0] = -1
    elif bad == "missing_role": del values[1]["exp_avg"]
    else: values[1]["gradient"] = values[1]["gradient"].astype(np.float64)
    with pytest.raises(EvidenceError): audit.audit_tensor(values)


def sealed_plan(tmp_path):
    captures = captured_sequence()
    contract = dict(input="fixed", initial_state="fresh", objective="fixed-adamw")
    configuration = {"runner": "synthetic:fixture"}
    training = {"steps": 3, "optimizer": audit.TARGET_OPTIMIZER, "clip_grad": 0}
    spec = {"schema_version": 1, "name": "independent-replay-reader-test", "contract": contract,
            "steps": [0, 1, 2], "stages": {
                "reference": {"role": "reference", "world_size": 2, "configuration": configuration},
                "candidate": {"role": "candidate", "world_size": 2, "configuration": configuration}},
            "tensors": {r + "/weight": {"shape": [6], "group": "test", "role": r} for r in audit.ROLES},
            "policy": [], "suite": {"training": training,
                                     "description": {"parameters": {"weight": {"shape": [6], "group": "test"}}}}}
    path = prepare(spec, tmp_path / "plan")
    plan = load_plan(path)
    for rank in range(2):
        with RankWriter(path, "reference", rank) as writer:
            # Alternate owners, reverse physical order: a reader must follow
            # global intervals, not concatenate physical file offsets.
            indices = list(range(rank, 6, 2))[::-1]
            for step in range(3):
                for role in audit.ROLES:
                    for index in indices:
                        writer.write(step, role + "/weight", captures[step][role][index:index+1], start=index)
            writer.finish(observed_contract=contract, configuration=configuration,
                          completed_steps=plan["steps"])
    finalize_stage(path, "reference", returncode=0)
    return path


def test_sharded_reader_verifies_all_roles_steps_and_rejects_corruption(tmp_path):
    path = sealed_plan(tmp_path)
    result = audit.audit_stage(path, "reference", expected_parameters=1, expected_steps=3)
    assert result["parameter_steps"] == 3 and result["continuity_checks"] == 2
    assert not result["invariant_failures"]
    assert all(v == 6 for r in result["rows"] for v in r["canonical_piece_counts"].values())
    raw = path.parent / "reference/rank1/float32.bin"
    content = raw.read_bytes()
    raw.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(EvidenceError, match="checksum"):
        audit.audit_stage(path, "reference")


def test_reject_nonzero_clipping_and_other_optimizer_contract():
    with pytest.raises(EvidenceError, match="clipping"):
        audit.validate_training({"steps": 3, "optimizer": audit.TARGET_OPTIMIZER, "clip_grad": 1})
    changed = {**audit.TARGET_OPTIMIZER, "lr": 1e-3}
    with pytest.raises(EvidenceError, match="scoped"):
        audit.validate_training({"steps": 3, "optimizer": changed, "clip_grad": 0})
