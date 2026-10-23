import math
import numpy as np
import pytest

from training_parity.metrics import Accumulator, compare_arrays
from training_parity.compare import fixed_check
from training_parity.common import EvidenceError


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-20, 1.0, 1e200, 1e300])
def test_scaled_norms_preserve_relative_error_across_fp64_range(scale):
    values = np.array([3.0, -4.0]) * scale
    result = compare_arrays(values, 2*values)
    assert result["relative_l2"] == pytest.approx(1.0)
    assert result["reference_l2"] == pytest.approx(5*scale, rel=1e-14, abs=0)
    assert result["absolute_l2"] == pytest.approx(5*scale, rel=1e-14, abs=0)
    assert result["cosine"] == pytest.approx(1.0)


def test_tiny_orthogonal_values_do_not_have_perfect_cosine():
    result = compare_arrays([1e-250, 0], [0, 1e-250])
    assert result["cosine"] == 0
    assert result["relative_l2"] == pytest.approx(math.sqrt(2))
    assert fixed_check({"metrics": result, "elementwise_violations": 0}, {"cosine_min": .999})


def test_merge_preserves_tiny_difference_beside_large_equal_values():
    left = np.array([1e200, 1e-200, -2e-200])
    right = np.array([1e200, 2e-200, -4e-200])
    merged = Accumulator()
    for x,y in zip(left,right):
        merged.merge(Accumulator().add([x],[y]))
    result = merged.result()
    assert result["absolute_l2"] == pytest.approx(math.sqrt(5)*1e-200, rel=1e-14, abs=0)
    direct = compare_arrays(left,right)
    for name in result:
        if result[name] is None:
            assert direct[name] is None
        else:
            assert result[name] == pytest.approx(direct[name], rel=1e-14, abs=0)
    assert result["relative_l2"] is None
    assert fixed_check({"metrics": result, "elementwise_violations": 0}, {"relative_l2": 0})


def test_only_exact_zero_has_zero_reference_semantics():
    assert compare_arrays([0.0], [0.0])["relative_l2"] == 0
    assert compare_arrays([0.0], [1e-300])["relative_l2"] is None
    assert compare_arrays([1e-300], [0.0])["relative_l2"] == 1
    assert compare_arrays([1e-300], [0.0])["cosine"] == 0


def test_unrepresentable_difference_fails_explicitly():
    with pytest.raises(EvidenceError, match="overflow"):
        compare_arrays([-1e308], [1e308])


def test_subnormal_rmse_cannot_satisfy_zero_error_gate():
    smallest = np.nextafter(0.0, 1.0)
    result = compare_arrays([0.0]*4, [smallest, 0.0, 0.0, 0.0])
    assert result["absolute_l2"] == smallest and result["rmse"] is None
    assert fixed_check({"metrics": result, "elementwise_violations": 0}, {"rmse": 0})


def test_subnormal_relative_error_avoids_rounded_absolute_norms():
    smallest = np.nextafter(0.0, 1.0)
    result = compare_arrays([2*smallest, 2*smallest], [3*smallest, 2*smallest])
    assert result["relative_l2"] == pytest.approx(1/(2*math.sqrt(2)))
    assert fixed_check({"metrics": result, "elementwise_violations": 0}, {"relative_l2": .34})
