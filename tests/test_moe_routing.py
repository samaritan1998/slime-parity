"""Hand-computed semantic cases, independent rational checks and fault injection."""
from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest

from slime_parity.diagnostics import moe_routing as audit


def bound(value=0., source='test-declaration-before-observations'):
    return audit.DeclaredBound(value, source)


def endpoint():
    x = np.array([[4, 4, 0, 0], [0, 0, 4, 4], [2, 2, 0, 0]], dtype=np.float32)
    route = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
    return audit.EndpointEvidence(label='synthetic-independent-endpoint',
        coordinates=((0, 0, 0), (0, 0, 1), (0, 0, 2)), expert_ids=(0, 1, 2, 3), inputs=x,
        weights=np.eye(4, dtype=np.float32), logits=x.copy(), routing_map=route,
        probabilities=np.where(route, np.float32(.5), np.float32(0)),
        contract=audit.RoutingContract(2, 'bfloat16', 'bfloat16', 'float32',
            'fp32_mul_add_or_fma', 'logits', 'selected_softmax'))


def dispatch(e=None):
    e = endpoint() if e is None else e
    row = np.array([0, 2, 0, 2, 1, 1], dtype=np.int64)
    expert = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    return audit.DispatchEvidence(row_token=row, row_expert=expert,
        expert_counts=np.array([2, 2, 1, 1]), dispatched_inputs=e.inputs[row].copy(),
        dispatched_probabilities=e.probabilities[row, expert].copy(), combine_token=row.copy(),
        contract=audit.DispatchContract('complete_no_drop_no_pad', 'expert_token', 'bfloat16', 'float32'))


def combine():
    return audit.CombineEvidence(
        expert_outputs=np.array([[1, 0], [10, 0], [0, 1], [0, 10], [1, 1], [2, 2]], dtype=np.float32),
        combined_output=np.array([[.5, .5], [1.5, 1.5], [5, 5]], dtype=np.float32),
        contract=audit.CombineContract('post_expert', 'float32', 'float32', 'float32', 'float32',
                                      'row_order', 'hand-specified arithmetic fixture'))


def routing(e=None):
    return audit.audit_endpoint(endpoint() if e is None else e,
                                probability_bound=bound(), probability_sum_bound=bound())


def test_hand_computed_full_local_catalog_pass_is_not_training_acceptance():
    result = audit.audit_layer(endpoint(), dispatch=dispatch(), combine=combine(),
        probability_bound=bound(), probability_sum_bound=bound(), combine_output_bound=bound())
    assert result['status'] == 'pass'
    assert result['training_acceptance'] is False
    assert result['distributed_matrix_qualification'] is False
    json.dumps(result, allow_nan=False)


def test_missing_dispatch_and_combine_cannot_disappear_from_catalog():
    result = audit.audit_layer(endpoint(), probability_bound=bound(), probability_sum_bound=bound())
    assert result['checks']['routing']['status'] == 'pass'
    assert result['status'] == 'inconclusive'
    assert result['checks']['dispatch']['status'] == result['checks']['combine']['status'] == 'inconclusive'


def test_probability_bound_is_not_invented_from_small_or_zero_residual():
    result = audit.audit_endpoint(endpoint())
    assert result['status'] == 'inconclusive'
    assert result['checks']['probabilities']['status'] == result['checks']['probability_sum']['status'] == 'inconclusive'


def test_analytic_dot_bound_covers_independent_exact_rational_dot_and_fp32_sum():
    rng = np.random.default_rng(811)
    x = audit.round_bfloat16(rng.normal(size=(3, 31)).astype(np.float32))
    w = audit.round_bfloat16(rng.normal(size=(4, 31)).astype(np.float32))
    p = audit.fp32_projection(x, w)
    assert p['domain'].all()
    for i in range(3):
        for j in range(4):
            exact = sum((Fraction(float(x[i, h])) * Fraction(float(w[j, h])) for h in range(31)), Fraction())
            actual = np.float32(0)
            for h in range(31):
                actual = np.float32(actual + np.float32(x[i, h] * w[j, h]))
            error = abs(Fraction(float(actual)) - exact)
            assert error <= Fraction(float(p['absolute_bound'][i, j]))
            assert abs(float(actual) - p['reference'][i, j]) <= p['absolute_bound'][i, j]
    assert p['assumptions']['fp32_unit_roundoff'] == 2.**-24
    assert p['assumptions']['fp64_unit_roundoff'] == 2.**-53
    assert p['assumptions']['operation_bound'] == 62
    assert not p['assumptions']['fp64_oracle_exact']


def test_projection_does_not_use_observed_logits_to_choose_bound():
    first = routing()
    e = endpoint(); wrong = e.logits.copy(); wrong[0, 0] += .25
    second = routing(replace(e, logits=wrong))
    assert first['checks']['projection']['max_bound'] == second['checks']['projection']['max_bound']
    assert second['checks']['projection']['status'] == 'fail'


@pytest.mark.parametrize('which', ['inputs', 'weights'])
def test_effective_bf16_exactness_checked_before_projection(which):
    e = endpoint(); value = getattr(e, which).copy(); value[0, 0] = np.float32(1.001)
    with pytest.raises(audit.EvidenceError, match='not exact effective BF16'):
        routing(replace(e, **{which: value}))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('which', ['inputs', 'weights', 'logits', 'probabilities'])
def test_nonfinite_evidence_rejected(which, value):
    e = endpoint(); array = getattr(e, which).copy(); array.flat[0] = value
    with pytest.raises(audit.EvidenceError, match='nonfinite'):
        routing(replace(e, **{which: array}))


def test_unknown_arithmetic_does_not_pass_even_for_identical_logits():
    e = endpoint()
    result = routing(replace(e, contract=replace(e.contract, projection_arithmetic='unknown_kernel')))
    assert result['status'] == result['same_input_stability']['status'] == 'inconclusive'


def test_float64_promotion_is_not_claimed_to_be_captured_fp32_storage():
    e = endpoint()
    with pytest.raises(audit.EvidenceError, match='storage differs'):
        routing(replace(e, logits=e.logits.astype(np.float64)))


@pytest.mark.parametrize('kind', ['operand_subnormal', 'product_subnormal', 'possible_cancel_subnormal', 'overflow'])
def test_fp32_analytic_domain_excludes_unproven_arithmetic(kind):
    if kind == 'operand_subnormal':
        x, w = np.float32(2.**-133), np.float32(1.)
    elif kind == 'product_subnormal':
        x, w = np.float32(2.**-100), np.float32(2.**-30)
    elif kind == 'possible_cancel_subnormal':
        x, w = np.float32(2.**-125), np.float32(1 + 2.**-7)
    else:
        x, w = np.float32(2.**100), np.float32(2.**100)
    p = audit.fp32_projection(np.array([[x]], dtype=np.float32), np.array([[w]], dtype=np.float32))
    assert not p['domain'].any()
    assert any(p['domain_exclusions'].values())


def test_small_normal_products_on_normal_lattice_remain_supported():
    p = audit.fp32_projection(np.array([[2.**-60, 2.**-60]], dtype=np.float32),
                              np.array([[2.**-60, -(2.**-60)]], dtype=np.float32))
    assert p['domain'].all() and p['reference'][0, 0] == 0


def test_zero_dot_has_zero_analytic_bound():
    p = audit.fp32_projection(np.zeros((2, 8), dtype=np.float32), np.ones((4, 8), dtype=np.float32))
    assert p['domain'].all() and not p['reference'].any() and not p['absolute_bound'].any()


def test_temporary_memory_blocking_preserves_entire_dot_and_bound(monkeypatch):
    rng = np.random.default_rng(812)
    x = audit.round_bfloat16(rng.normal(size=(5, 13)).astype(np.float32))
    w = audit.round_bfloat16(rng.normal(size=(7, 13)).astype(np.float32))
    whole = audit.fp32_projection(x, w)
    monkeypatch.setattr(audit, '_PROJECTION_PRODUCT_ELEMENTS', 27)
    blocked = audit.fp32_projection(x, w)
    for key in ('reference', 'absolute_bound', 'domain'):
        assert np.array_equal(whole[key], blocked[key])


@pytest.mark.parametrize('selected', [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
def test_exact_cutoff_tie_all_legal_sets_allowed_without_index_tiebreak(selected):
    scores = np.array([[2, 2, 2, 0]], dtype=np.float32)
    result = audit.legal_topk(scores, np.array([selected], dtype=bool), 2)
    assert result['status'] == 'pass' and result['cutoff_tie_tokens'] == 1
    stable = audit.topk_stability(scores, np.zeros_like(scores, dtype=np.float64), 2)
    assert not stable['stable'][0] and stable['report']['status'] == 'inconclusive'


@pytest.mark.parametrize('selected', [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]])
def test_wrong_topk_lower_score_missing_required_and_wrong_count_fail(selected):
    result = audit.legal_topk(np.array([[4, 3, 2, 1]], dtype=np.float32),
                             np.array([selected], dtype=bool), 2)
    assert result['status'] == 'fail'


def test_tie_is_exact_not_a_tolerance_bucket():
    scores = np.array([[2, 1, np.nextafter(np.float32(1), np.float32(0))]], dtype=np.float32)
    result = audit.legal_topk(scores, np.array([[1, 0, 1]], dtype=bool), 2)
    assert result['status'] == 'fail' and result['cutoff_tie_tokens'] == 0


def test_captured_slot_conversion_preserves_expert_probability_alignment():
    selected = np.array([[3, 1], [0, 2]])
    probabilities = np.array([[.25, .75], [.125, .875]], dtype=np.float32)
    route, dense = audit.routing_from_slots(selected, probabilities, 4)
    assert np.array_equal(route, [[False, True, False, True], [True, False, True, False]])
    assert np.array_equal(dense, [[0, .75, 0, .25], [.125, 0, .875, 0]])
    reverse_route, reverse_dense = audit.routing_from_slots(selected[:, ::-1], probabilities[:, ::-1], 4)
    assert np.array_equal(route, reverse_route) and np.array_equal(dense, reverse_dense)


@pytest.mark.parametrize('indices', [[[0, 0]], [[0, -1]], [[0, 4]]])
def test_bad_observed_topk_slots_are_not_silently_coalesced(indices):
    with pytest.raises(audit.EvidenceError, match='Duplicate|out of range'):
        audit.routing_from_slots(np.array(indices), np.array([[.5, .5]], dtype=np.float32), 4)


@pytest.mark.parametrize('error, stable', [(.499, True), (.5, False), (.501, False)])
def test_strict_interval_margin_boundary(error, stable):
    value = np.array([[3., 2., 1., 0.]], dtype=np.float64)
    result = audit.topk_stability(value, np.full_like(value, error), 2)
    assert result['stable'].tolist() == [stable]


def test_per_expert_asymmetric_bound_used_instead_of_single_cutoff_margin():
    scores = np.array([[5, 4, 0, -1]], dtype=np.float64)
    assert audit.topk_stability(scores, np.array([[0, 0, 5, 0]], dtype=np.float64), 2)['stable'].tolist() == [False]


def test_all_experts_topk_is_set_stable_without_nan_margin():
    result = audit.topk_stability(np.array([[1., 1.]]), np.zeros((1, 2)), 2)
    assert result['stable'].tolist() == [True]
    json.dumps(result['report'], allow_nan=False)


def test_equal_routes_without_external_upstream_budget_still_inconclusive():
    result = audit.audit_pair(endpoint(), replace(endpoint(), label='candidate'))
    assert result['status'] == 'inconclusive' and result['route_set_mismatch_tokens'] == 0
    assert not result['budget_estimated_from_observed_difference']


def test_declared_cross_budget_qualifies_only_local_route_stability():
    result = audit.audit_pair(endpoint(), replace(endpoint(), label='candidate'), cross_end_bound=bound(.01))
    assert result['status'] == 'pass' and not result['training_acceptance']


def test_upstream_difference_is_not_excused_by_each_ends_own_projection_bound():
    e = endpoint(); x = e.inputs.copy(); x[0, 0] += .25
    c = replace(e, label='candidate', inputs=x, logits=x.copy())
    result = audit.audit_pair(e, c, cross_end_bound=bound(.1))
    assert result['checks']['reference_projection']['status'] == result['checks']['candidate_projection']['status'] == 'pass'
    assert result['checks']['cross_end_budget']['status'] == result['status'] == 'fail'


def test_candidate_wrong_effective_weight_detected_independently():
    e = endpoint(); w = e.weights.copy(); w[0, 0] += .25
    result = audit.audit_pair(e, replace(e, label='candidate', weights=w), cross_end_bound=bound(10.))
    assert result['checks']['candidate_projection']['status'] == 'fail'


def test_known_local_failure_is_retained_when_cross_budget_is_missing():
    e = endpoint(); w = e.weights.copy(); w[0, 0] += .25
    result = audit.audit_pair(e, replace(e, label='candidate', weights=w))
    assert result['checks']['cross_end_stability']['status'] == 'inconclusive'
    assert result['checks']['candidate_projection']['status'] == result['status'] == 'fail'


def test_cross_bound_large_enough_for_observed_routes_does_not_make_ties_stable():
    result = audit.audit_pair(endpoint(), replace(endpoint(), label='candidate'), cross_end_bound=bound(3.))
    assert result['checks']['cross_end_budget']['status'] == 'pass'
    assert result['status'] == 'inconclusive'


def test_cross_token_coordinates_must_match_even_with_equal_values():
    e = endpoint()
    with pytest.raises(audit.EvidenceError, match='identities differ'):
        audit.audit_pair(e, replace(e, coordinates=tuple(reversed(e.coordinates))), cross_end_bound=bound(.1))


def test_cross_expert_identity_order_cannot_be_replaced_by_equal_dimension():
    e = endpoint()
    with pytest.raises(audit.EvidenceError, match='identities differ'):
        audit.audit_pair(e, replace(e, expert_ids=(1, 0, 2, 3)), cross_end_bound=bound(.1))


@pytest.mark.parametrize('change', ['wrong_values_sum_one', 'outside_support', 'negative', 'wrong_sum'])
def test_probabilities_checked_beyond_row_sum(change):
    e = endpoint(); p = e.probabilities.copy()
    if change == 'wrong_values_sum_one':
        p[0, 0], p[0, 1] = .25, .75
    elif change == 'outside_support':
        p[0, 1], p[0, 2] = 0., .5
    elif change == 'negative':
        p[0, 0], p[0, 1] = -.5, 1.5
    else:
        p[0, 0] = .25
    assert routing(replace(e, probabilities=p))['status'] == 'fail'


def test_topk_slot_probability_swap_detected_despite_identical_sum_and_routes():
    e = endpoint(); x = e.inputs.copy(); x[0, 0] = 5
    correct = np.float32(1. / (1. + np.exp(-1.)))
    p = e.probabilities.copy(); p[0, 0], p[0, 1] = np.float32(1 - correct), correct
    result = audit.audit_endpoint(replace(e, inputs=x, logits=x.copy(), probabilities=p),
                                 probability_bound=bound(1e-7), probability_sum_bound=bound(1e-7))
    assert result['checks']['topk']['status'] == result['checks']['probability_sum']['status'] == 'pass'
    assert result['checks']['probabilities']['status'] == 'fail'


def test_full_softmax_selection_requires_actual_score_capture_and_separate_bound():
    e = endpoint(); c = replace(e.contract, selection_domain='full_softmax', probability_rule='full_softmax_selected_renorm')
    missing = replace(e, contract=c)
    assert routing(missing)['status'] == 'inconclusive'
    actual = replace(missing, selection_scores=audit.softmax64(e.logits).astype(np.float32))
    assert routing(actual)['status'] == 'inconclusive'
    result = audit.audit_endpoint(actual, probability_bound=bound(1e-7), probability_sum_bound=bound(1e-7),
                                 selection_score_bound=bound(1e-7))
    assert result['status'] == 'pass'
    assert result['same_input_stability']['status'] == 'inconclusive'


def test_reported_projection_subnormal_output_is_excluded_not_excused_by_bound():
    e = endpoint(); logits = e.logits.copy(); logits[0, 2] = np.float32(2.**-133)
    result = routing(replace(e, logits=logits))
    assert result['checks']['projection']['status'] == 'inconclusive'
    assert result['checks']['projection']['excluded_elements'] == 1
    paired = audit.audit_pair(e, replace(e, logits=logits), cross_end_bound=bound(.1))
    assert paired['checks']['candidate_projection']['status'] == paired['status'] == 'inconclusive'


@pytest.mark.parametrize('kind', ['empty', 'weight_shape', 'route_dtype', 'prob_shape', 'coordinate_duplicate'])
def test_malformed_or_empty_evidence_fails_closed(kind):
    e = endpoint()
    if kind == 'empty':
        e = replace(e, inputs=e.inputs[:0])
    elif kind == 'weight_shape':
        e = replace(e, weights=e.weights[:, :2])
    elif kind == 'route_dtype':
        e = replace(e, routing_map=e.routing_map.astype(np.int32))
    elif kind == 'prob_shape':
        e = replace(e, probabilities=e.probabilities[:, :2])
    else:
        e = replace(e, coordinates=(e.coordinates[0],) * 3)
    with pytest.raises(audit.EvidenceError):
        routing(e)


def test_dispatch_missing_row_is_not_accepted_as_partial_coverage():
    d = dispatch()
    with pytest.raises(audit.EvidenceError, match='exact shape'):
        audit.audit_dispatch(endpoint(), replace(d, row_token=d.row_token[:-1]))


def test_full_softmax_outside_support_fails_even_with_large_declared_error():
    e = endpoint(); scores = audit.softmax64(e.logits).astype(np.float32); scores[0, 0] = -1
    e = replace(e, selection_scores=scores, contract=replace(e.contract, selection_domain='full_softmax',
                probability_rule='full_softmax_selected_renorm'))
    result = audit.audit_endpoint(e, probability_bound=bound(10), probability_sum_bound=bound(10),
                                 selection_score_bound=bound(10))
    assert result['checks']['selection_scores']['status'] == 'fail'


@pytest.mark.parametrize('kind', ['missing_pair', 'wrong_expert', 'wrong_inverse', 'counts', 'input', 'probability', 'order'])
def test_actual_dispatch_faults_are_detected(kind):
    d = dispatch()
    if kind == 'missing_pair':
        row = d.row_token.copy(); row[1] = row[0]; d = replace(d, row_token=row)
    elif kind == 'wrong_expert':
        expert = d.row_expert.copy(); expert[0] = 3; d = replace(d, row_expert=expert)
    elif kind == 'wrong_inverse':
        inv = d.combine_token.copy(); inv[0] = 1; d = replace(d, combine_token=inv)
    elif kind == 'counts':
        counts = d.expert_counts.copy(); counts[0] += 1; d = replace(d, expert_counts=counts)
    elif kind == 'input':
        values = d.dispatched_inputs.copy(); values[0, 0] += 1; d = replace(d, dispatched_inputs=values)
    elif kind == 'probability':
        values = d.dispatched_probabilities.copy(); values[0] = .25; d = replace(d, dispatched_probabilities=values)
    else:
        idx = np.array([1, 0, 2, 3, 4, 5])
        d = replace(d, row_token=d.row_token[idx], row_expert=d.row_expert[idx],
                    combine_token=d.combine_token[idx], dispatched_inputs=d.dispatched_inputs[idx],
                    dispatched_probabilities=d.dispatched_probabilities[idx])
    assert audit.audit_dispatch(endpoint(), d)['status'] == 'fail'


def test_transport_signed_zero_is_not_silently_equal():
    d = dispatch(); values = d.dispatched_inputs.copy(); values[0, 2] = np.float32(-0.)
    assert audit.audit_dispatch(endpoint(), replace(d, dispatched_inputs=values))['checks']['inputs']['status'] == 'fail'


def test_missing_inverse_is_inconclusive_even_with_correct_pair_counts():
    assert audit.audit_dispatch(endpoint(), replace(dispatch(), combine_token=None))['status'] == 'inconclusive'


@pytest.mark.parametrize('kind', ['unknown', 'distributed_shard', 'capacity_padding'])
def test_unsupported_dispatch_domain_is_never_promoted_by_good_counts(kind):
    d = dispatch(); d = replace(d, contract=replace(d.contract, domain=kind))
    assert audit.audit_dispatch(endpoint(), d)['status'] == 'inconclusive'


def test_expert_grouped_any_order_does_not_assume_token_sort():
    d = dispatch(); idx = np.array([1, 0, 2, 3, 4, 5])
    d = replace(d, row_token=d.row_token[idx], row_expert=d.row_expert[idx], combine_token=d.combine_token[idx],
                dispatched_inputs=d.dispatched_inputs[idx], dispatched_probabilities=d.dispatched_probabilities[idx],
                contract=replace(d.contract, order='expert_grouped_any'))
    assert audit.audit_dispatch(endpoint(), d)['status'] == 'pass'


@pytest.mark.parametrize('kind', ['omitted_contribution', 'duplicate_contribution', 'unweighted_sum', 'wrong_token'])
def test_weighted_combine_mistakes_detected(kind):
    c = combine(); actual = c.combined_output.copy()
    if kind == 'omitted_contribution':
        actual[0, 1] = 0
    elif kind == 'duplicate_contribution':
        actual[0, 1] = 1
    elif kind == 'unweighted_sum':
        actual *= 2
    else:
        actual[[0, 1]] = actual[[1, 0]]
    result = audit.audit_combine(endpoint(), dispatch(), replace(c, combined_output=actual), output_bound=bound())
    assert result['status'] == 'fail'


def test_preweighted_output_cannot_prove_internal_probability_application():
    c = combine(); c = replace(c, expert_outputs=c.expert_outputs * np.float32(.5),
        contract=replace(c.contract, weighting='preweighted', product_dtype='already_weighted'))
    result = audit.audit_combine(endpoint(), dispatch(), c, output_bound=bound())
    assert result['checks']['combine']['status'] == 'pass'
    assert result['checks']['probability_weighting']['status'] == result['status'] == 'inconclusive'


@pytest.mark.parametrize('field,value', [('accumulation_order', 'unknown_atomic_order'), ('product_dtype', 'unknown'),
                                      ('source', ''), ('weighting', 'unknown')])
def test_unknown_combine_arithmetic_does_not_pass_on_matching_output(field, value):
    c = combine(); c = replace(c, contract=replace(c.contract, **{field: value}))
    assert audit.audit_combine(endpoint(), dispatch(), c, output_bound=bound())['status'] == 'inconclusive'


def test_no_combine_tolerance_is_manufactured_from_matching_output():
    assert audit.audit_combine(endpoint(), dispatch(), combine())['status'] == 'inconclusive'


@pytest.mark.parametrize('value', [-1, np.inf, np.nan, True, np.array([0., 0.])])
def test_invalid_bound_rejected(value):
    with pytest.raises(audit.EvidenceError, match='Bound|bound'):
        bound(value).array((3, 4))


def test_no_bound_declaration_identity_is_rejected():
    with pytest.raises(audit.EvidenceError, match='declaration identity'):
        audit.DeclaredBound(1., '').array((1,))


def test_bf16_rounding_ties_to_even_and_signed_zero():
    # Adjacent FP32 words exactly halfway between BF16 numbers, even/odd LSB.
    inputs = np.array([0x3f808000, 0x3f818000, 0x80000000], dtype=np.uint32).view(np.float32)
    expected = np.array([0x3f800000, 0x3f820000, 0x80000000], dtype=np.uint32)
    assert np.array_equal(audit.round_bfloat16(inputs).view(np.uint32), expected)


def test_calls_do_not_mutate_runtime_evidence():
    e, d, c = endpoint(), dispatch(), combine()
    arrays = [x for obj in (e, d, c) for x in vars(obj).values() if isinstance(x, np.ndarray)]
    before = [x.tobytes() for x in arrays]
    audit.audit_layer(e, dispatch=d, combine=c, probability_bound=bound(),
                      probability_sum_bound=bound(), combine_output_bound=bound())
    audit.audit_pair(e, replace(e, label='candidate'), cross_end_bound=bound(.01))
    assert [x.tobytes() for x in arrays] == before


def test_module_imports_no_training_backend_or_reference():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(modules) <= {'dataclasses', 'typing', 'numpy'}


# Zero-hit experts remain in the identity/count domain without fabricated rows.
def unobserved_experts_fixture():
    original = endpoint()
    inputs = np.array([[4, 4, 0, 0], [2, 2, 0, 0], [8, 8, 0, 0]], dtype=np.float32)
    route = np.array([[True, True, False, False]] * 3)
    e = replace(original, inputs=inputs, logits=inputs.copy(), routing_map=route,
                probabilities=np.where(route, np.float32(.5), np.float32(0)))
    rows = np.array([0, 1, 2, 0, 1, 2]); experts = np.array([0, 0, 0, 1, 1, 1])
    d = audit.DispatchEvidence(rows, experts, np.array([3, 3, 0, 0]), inputs[rows].copy(),
        np.full(6, .5, dtype=np.float32), rows.copy(),
        audit.DispatchContract('complete_no_drop_no_pad', 'expert_token', 'bfloat16', 'float32'))
    return e, d


def test_zero_hit_experts_need_no_synthetic_rows_or_outputs():
    e, d = unobserved_experts_fixture()
    result = audit.audit_dispatch(e, d)
    assert result['status'] == 'pass'
    assert result['checks']['pair_permutation']['rows'] == 6
    assert np.array_equal(d.expert_counts, [3, 3, 0, 0])
    assert not (d.row_expert >= 2).any()
    assert audit.audit_endpoint(e, probability_bound=bound(), probability_sum_bound=bound())['status'] == 'pass'


def test_unobserved_expert_count_tail_cannot_be_omitted():
    e, d = unobserved_experts_fixture()
    with pytest.raises(audit.EvidenceError, match='expert_counts.*exact shape'):
        audit.audit_dispatch(e, replace(d, expert_counts=np.array([3, 3])))


def test_fabricated_empty_expert_row_is_not_justified_by_matching_counts():
    e, d = unobserved_experts_fixture()
    experts = d.row_expert.copy(); experts[-1] = 2
    result = audit.audit_dispatch(e, replace(d, row_expert=experts, expert_counts=np.array([3, 2, 1, 0])))
    assert result['checks']['expert_counts']['status'] == 'pass'
    assert result['checks']['pair_permutation']['status'] == result['status'] == 'fail'


def test_empty_expert_observation_does_not_create_an_expert_correctness_claim():
    e, d = unobserved_experts_fixture()
    result = audit.audit_layer(e, dispatch=d, probability_bound=bound(), probability_sum_bound=bound())
    assert result['checks']['dispatch']['status'] == 'pass'
    assert result['checks']['combine']['status'] == result['status'] == 'inconclusive'
    assert not result['training_acceptance']
