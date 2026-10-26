from decimal import Decimal, localcontext

import numpy as np
import pytest

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

original = _load_diagnostic("adamw_replay_audit")
audit = _load_diagnostic("adamw_hyperparam_replay")
from training_parity.common import EvidenceError


def test_exact_binary_scalar_complements():
    values = audit.scalars(1)
    assert float(values["beta1"]) == 7549747 / 8388608
    assert float(values["beta2"]) == 16760439 / 16777216
    assert float(values["one_minus_beta1"]) == 838861 / 8388608
    assert float(values["one_minus_beta2"]) == 16777 / 16777216
    assert values["one_minus_beta1"] != np.float32(1 - .9)
    assert values["one_minus_beta2"] != np.float32(1 - .999)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_host_bias_correction_against_exact_decimal_power(step):
    values = audit.scalars(step)
    with localcontext() as context:
        context.prec = 80
        for number in (1, 2):
            beta = Decimal.from_float(float(values[f"beta{number}"]))
            expected = np.float32(float(Decimal(1) - beta ** step))
            assert expected.tobytes() == values[f"bias_correction{number}"].tobytes()


def test_fp64_moments_against_decimal_for_nonzero_previous_state():
    p, g, m, v = [np.array([value], dtype=np.float32) for value in (.25, .03125, -.125, .0625)]
    result = audit.formula_step(p, g, m, v, 2, np.float64)
    scalars = audit.scalars(2)
    with localcontext() as context:
        context.prec = 80
        d = lambda x: Decimal.from_float(float(x))
        expected_m = d(scalars["beta1"]) * d(m[0]) + d(scalars["one_minus_beta1"]) * d(g[0])
        expected_v = d(scalars["beta2"]) * d(v[0]) + d(scalars["one_minus_beta2"]) * d(g[0]) ** 2
    assert result["exp_avg"][0] == float(expected_m)
    assert result["exp_avg_sq"][0] == float(expected_v)


def test_first_step_separates_hyperparameter_hypotheses_without_mutating_inputs():
    p = np.array([1., 0., -.25], dtype=np.float32)
    g = np.array([.03125, 1e-10, -.125], dtype=np.float32)
    z = np.zeros_like(p)
    before = [x.tobytes() for x in (p, g, z)]
    result = audit.formula_step(p, g, z, z, 1, np.float64)
    prior = original.formula_step(p, g, z, z, 1, np.float64)
    assert np.all(result["exp_avg_sq"] < prior["exp_avg_sq"])
    assert np.allclose(result["exp_avg_sq"] / prior["exp_avg_sq"],
                       (16777 / 16777216) / (1 - .999), rtol=1e-15, atol=0)
    assert [x.tobytes() for x in (p, g, z)] == before
    for precision in (np.float32, np.float64):
        result = audit.formula_step(p, g, z, z, 1, precision)
        np.testing.assert_array_equal(result["update"],
                                      np.subtract(result["parameter"], p, dtype=np.float32))


def test_nonfinite_negative_variance_and_unsupported_steps_rejected():
    x = np.ones(2, dtype=np.float32)
    with pytest.raises(EvidenceError, match="Nonfinite"):
        audit.formula_step(x, np.array([np.nan, 0], dtype=np.float32), x, x, 1, np.float32)
    with pytest.raises(EvidenceError, match="Negative"):
        audit.formula_step(x, x, x, -x, 1, np.float32)
    with pytest.raises(EvidenceError, match="three supported steps"):
        audit.formula_step(x, x, x, x, 4, np.float32)


def test_sharded_replay_accepts_catalog_count_and_preserves_prior_audit(tmp_path):
    from copy import deepcopy
    from test_adamw_replay_diagnostic import sealed_plan

    path = sealed_plan(tmp_path)
    baseline = original.audit_stage(path, "reference", expected_parameters=1, expected_steps=3)
    unchanged = deepcopy(baseline)
    result = audit.stage_report(path, "reference", baseline)
    assert baseline == unchanged
    assert result["parameters"] == 1 and result["parameter_steps"] == 3
    assert not result["invariant_failures"] and result["capture_seals_verified"]
    for variant in audit.VARIANTS:
        assert result["formula_summary"][variant]["exp_avg_sq"]["tensor_steps"] == 3
    changed = deepcopy(baseline)
    changed["raw_files"]["rank0/float32.bin"]["sha256"] = "0" * 64
    with pytest.raises(EvidenceError, match="capture differs"):
        audit.stage_report(path, "reference", changed)
