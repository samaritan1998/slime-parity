"""Independent, local MoE routing diagnostics using only NumPy and stdlib.

An endpoint supplies its own observed inputs, effective weights and outputs.
Neither endpoint supplies expected routes for the other. ``pass`` means only
that a named local check meets its stated contract; it is never training
acceptance. Missing evidence, unsupported arithmetic and absent independently
declared budgets remain ``inconclusive``. No budget is estimated from residuals.

The complete-row dispatch API describes one logical no-drop/no-pad routing
domain. It does not establish distributed ownership, padding or replica cover.
Such observations must first be independently reconstructed and sealed by a
caller. Counts or reconstructed arrays are not substitutes for runtime capture.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


_PROJECTION_PRODUCT_ELEMENTS = 1 << 20


class EvidenceError(ValueError):
    """Malformed or internally inconsistent evidence; never a passing check."""


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


@dataclass(frozen=True)
class DeclaredBound:
    """Absolute error envelope declared externally, before seeing residuals.

    ``absolute`` is a scalar or exactly the compared array's shape. ``source``
    identifies its frozen derivation/decision, not this run's measured error.
    This class cannot certify an external declaration's scientific adequacy.
    """
    absolute: object
    source: str

    def array(self, shape):
        _require(isinstance(self.source, str) and bool(self.source.strip()),
                 'Bound needs an external declaration identity')
        value = np.asarray(self.absolute)
        _require(value.dtype.kind in 'fiu' and value.dtype.kind != 'b', 'Invalid bound dtype')
        value = value.astype(np.float64)
        _require(value.shape in ((), tuple(shape)) and np.isfinite(value).all()
                 and (value >= 0).all(), 'Bound must be finite, nonnegative and scalar or exact shape')
        return np.broadcast_to(value, shape)


@dataclass(frozen=True)
class RoutingContract:
    top_k: int
    input_dtype: str
    weight_dtype: str
    logit_dtype: str
    projection_arithmetic: str
    selection_domain: str
    probability_rule: str


@dataclass(frozen=True)
class EndpointEvidence:
    label: str
    coordinates: tuple
    expert_ids: tuple
    inputs: np.ndarray
    weights: np.ndarray
    logits: np.ndarray
    routing_map: np.ndarray
    probabilities: np.ndarray
    contract: RoutingContract
    selection_scores: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DispatchContract:
    domain: str
    order: str
    input_dtype: str
    probability_dtype: str


@dataclass(frozen=True)
class DispatchEvidence:
    """All arrays here are observations, not analyzer-generated expected data.

    row_token/row_expert identify the actual dispatched row. combine_token is
    the actual inverse scatter destination (tokens occur K times), not an
    invented bijective inverse over the unexpanded token domain.
    """
    row_token: np.ndarray
    row_expert: np.ndarray
    expert_counts: np.ndarray
    dispatched_inputs: np.ndarray
    dispatched_probabilities: np.ndarray
    combine_token: Optional[np.ndarray]
    contract: DispatchContract


@dataclass(frozen=True)
class CombineContract:
    weighting: str
    expert_output_dtype: str
    product_dtype: str
    accumulation_dtype: str
    output_dtype: str
    accumulation_order: str
    source: str


@dataclass(frozen=True)
class CombineEvidence:
    expert_outputs: np.ndarray
    combined_output: np.ndarray
    contract: CombineContract


def _result(status, reason, **details):
    return dict(status=status, reason=reason, **details)


def _aggregate(checks):
    states = [check['status'] for check in checks.values()]
    _require(states and set(states) <= {'pass', 'fail', 'inconclusive'}, 'Empty/unknown check catalog')
    return 'fail' if 'fail' in states else 'inconclusive' if 'inconclusive' in states else 'pass'


def _float_array(value, name, ndim=None):
    _require(isinstance(value, np.ndarray) and value.dtype in (np.dtype('float32'), np.dtype('float64')),
             name + ' must be an explicit float32/float64 ndarray')
    _require((ndim is None or value.ndim == ndim) and value.size > 0
             and np.isfinite(value).all(), name + ' has invalid dimensions, empty or nonfinite data')
    return value


def _integer_array(value, name, shape):
    _require(isinstance(value, np.ndarray) and value.dtype.kind in 'iu'
             and value.shape == tuple(shape), name + ' must be an integer array of the exact shape')
    return value


def _effective_dtype(value, declared, name):
    """BF16 is represented by exact promoted float32 values, not ml_dtypes."""
    _float_array(value, name)
    if declared == 'bfloat16':
        _require(value.dtype == np.float32, name + ' BF16 evidence must be promoted float32')
        bits = np.ascontiguousarray(value).view(np.uint32)
        _require(np.all((bits & np.uint32(0xffff)) == 0), name + ' is not exact effective BF16')
    elif declared == 'float32':
        _require(value.dtype == np.float32, name + ' declared float32 but storage differs')
    elif declared == 'float64':
        _require(value.dtype == np.float64, name + ' declared float64 but storage differs')
    else:
        return False
    return True


def round_bfloat16(value):
    """Round float32 to BF16, ties to even, returning exact float32 promotions.

    The input conversion is explicit: this is FP32 -> BF16, not an assertion
    that an arbitrary FP64 -> BF16 cast has no intermediate rounding.
    """
    x = np.asarray(value, dtype=np.float32)
    _require(np.isfinite(x).all(), 'Cannot BF16-round nonfinite/overflowed inputs')
    bits = np.ascontiguousarray(x).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)
    result = rounded.view(np.float32).reshape(x.shape)
    _require(np.isfinite(result).all(), 'BF16 rounding overflow is outside this contract')
    return result


def _round(value, dtype):
    if dtype == 'bfloat16':
        return round_bfloat16(value)
    _require(dtype in ('float32', 'float64'), 'Unsupported explicit rounding dtype')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(value, dtype=dtype)
    _require(np.isfinite(result).all(), 'Explicit arithmetic overflow is outside this contract')
    return result


def _up(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value == 0, 0., np.nextafter(value, np.inf))


def _gamma(operations, precision):
    unit = 2. ** (-precision)
    _require(type(operations) is int and operations >= 0 and operations * unit < 1,
             'Dot length is outside the gamma bound domain')
    return float(_up((operations * unit) / (1. - operations * unit)))


def fp32_projection(inputs, weights):
    """Independent FP64 projection and conservative ordinary FP32 dot bound.

    Domain: exact promoted BF16 operands; round-to-nearest ordinary FP32 multiply/add or FMA
    reduction, at most 2H roundings; no intermediate overflow; no nonzero
    subnormal operands/products/sums. BF16 products have <=16 significant bits
    and are exact in FP64. Requiring every product to be an integer multiple
    of 2^-126 conservatively excludes cancellation that could enter the FP32
    subnormal domain in any accumulation order. This can reject valid kernels;
    rejection yields inconclusive, not a larger budget.

    gamma_(2H,24)*sumabs bounds FP32 error. A separately upward-rounded
    gamma_(2H,53)*sumabs bounds this FP64 sum, and the sumabs estimate is itself
    inflated by its FP64 bound. No observed logit or endpoint difference is
    used to construct any part of this envelope. Kernel compliance with the
    stated ordinary-arithmetic model remains a declared premise.
    """
    _float_array(inputs, 'inputs', 2)
    _float_array(weights, 'weights', 2)
    _require(inputs.shape[1] == weights.shape[1], 'Projection dimensions differ')
    _effective_dtype(inputs, 'bfloat16', 'inputs')
    _effective_dtype(weights, 'bfloat16', 'weights')
    width = inputs.shape[1]
    g32, g64 = _gamma(2 * width, 24), _gamma(2 * width, 53)
    shape = (inputs.shape[0], weights.shape[0])
    reference, sumabs = np.empty(shape), np.empty(shape)
    lattice_domain = np.empty(shape, dtype=bool)
    expert_block = min(weights.shape[0], max(1, _PROJECTION_PRODUCT_ELEMENTS // width))
    token_block = max(1, _PROJECTION_PRODUCT_ELEMENTS // (width * expert_block))
    # Bound temporary T*E*H storage; block boundaries never split a dot product.
    for t in range(0, inputs.shape[0], token_block):
        for e in range(0, weights.shape[0], expert_block):
            section = (slice(t, t + token_block), slice(e, e + expert_block))
            products = (inputs[t:t + token_block].astype(np.float64)[:, None, :]
                        * weights[e:e + expert_block].astype(np.float64)[None, :, :])
            reference[section] = products.sum(axis=2, dtype=np.float64)
            sumabs[section] = np.abs(products).sum(axis=2, dtype=np.float64)
            # Scaling binary BF16 products by 2^126 is exact and fits FP64.
            scaled = np.ldexp(products, 126)
            lattice_domain[section] = (scaled == np.trunc(scaled)).all(axis=2)
    upper_sumabs = _up(sumabs / (1. - g64))
    fp32_error, fp64_error = _up(g32 * upper_sumabs), _up(g64 * upper_sumabs)
    bound = _up(fp32_error + fp64_error)
    tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    x = np.abs(inputs.astype(np.float64)); w = np.abs(weights.astype(np.float64))
    operand_domain = (((x == 0) | (x >= tiny)).all(axis=1)[:, None]
                      & ((w == 0) | (w >= tiny)).all(axis=1)[None, :])
    overflow_domain = _up(upper_sumabs * (1. + g32)) <= maximum
    domain = operand_domain & lattice_domain & overflow_domain
    return dict(reference=reference, absolute_bound=bound, domain=domain,
        domain_exclusions=dict(non_normal_operand=int((~operand_domain).sum()),
            possible_subnormal_product_or_partial_sum=int((~lattice_domain).sum()),
            possible_fp32_overflow=int((~overflow_domain).sum())),
        assumptions=dict(input_and_weight='effective BF16, exact float32 promotion',
            projection='ordinary FP32 multiply/add or FMA, at most 2H roundings',
            rounding='round to nearest, ties to even',
            fp32_unit_roundoff=2.**-24, fp64_unit_roundoff=2.**-53,
            operation_bound=2 * width, gamma_fp32=g32, gamma_fp64=g64,
            intermediate_overflow_excluded=True, subnormal_domain_conservatively_excluded=True,
            kernel_instructions_observed=False, fp64_oracle_exact=False))


def _comparison(expected, observed, bound, source):
    _require(expected.shape == observed.shape and np.isfinite(expected).all()
             and np.isfinite(observed).all(), 'Compared arrays must be finite with identical shapes')
    error = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
    violating = error > bound
    return _result('fail' if violating.any() else 'pass', 'declared_absolute_error_envelope',
        compared_elements=int(error.size), different_elements=int((error != 0).sum()),
        violating_elements=int(violating.sum()), max_abs=float(error.max()),
        max_bound=float(np.max(bound)), bound_source=source)


def _declared_comparison(expected, observed, bound, missing_reason):
    if bound is None:
        return _result('inconclusive', missing_reason)
    return _comparison(expected, observed, bound.array(expected.shape), bound.source)


def softmax64(values):
    _float_array(values, 'softmax values', 2)
    shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
    with np.errstate(under='ignore'):
        exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def legal_topk(scores, routing_map, top_k):
    """Check membership, permitting any exact cutoff tie (no index tie-break)."""
    _float_array(scores, 'selection scores', 2)
    _require(isinstance(routing_map, np.ndarray) and routing_map.dtype == np.bool_
             and routing_map.shape == scores.shape, 'Routing map dimensions/dtype differ')
    _require(type(top_k) is int and 0 < top_k <= scores.shape[1], 'Invalid top_k')
    cut = np.partition(scores, scores.shape[1] - top_k, axis=1)[:, -top_k]
    above, below = scores > cut[:, None], scores < cut[:, None]
    counts = routing_map.sum(axis=1)
    valid = ((counts == top_k) & ~(above & ~routing_map).any(axis=1)
             & ~(below & routing_map).any(axis=1))
    at_cut = scores == cut[:, None]
    ties = at_cut.sum(axis=1) > (top_k - above.sum(axis=1))
    return _result('pass' if valid.all() else 'fail', 'exact_selection_score_topk_membership',
        tokens=int(len(valid)), invalid_tokens=int((~valid).sum()),
        cutoff_tie_tokens=int(ties.sum()), valid_tokens=valid.tolist(),
        tie_policy='any subset of exact cutoff ties, all strictly larger scores required')


def routing_from_slots(selected_experts, selected_probabilities, num_experts):
    """Convert captured slots to derived sparse matrices, preserving own values.

    The outputs are explicitly derived representation, not new observations of
    native slots or dispatch. Duplicate/out-of-range observed slots are rejected.
    """
    _require(type(num_experts) is int and num_experts > 0, 'Invalid expert count')
    _require(isinstance(selected_experts, np.ndarray) and selected_experts.ndim == 2
             and selected_experts.size > 0 and selected_experts.dtype.kind in 'iu',
             'Captured selected experts must be a nonempty integer matrix')
    t, k = selected_experts.shape
    _require(0 < k <= num_experts and ((0 <= selected_experts) & (selected_experts < num_experts)).all(),
             'Captured expert slot out of range')
    _require(not (np.diff(np.sort(selected_experts, axis=1), axis=1) == 0).any(),
             'Duplicate captured expert slot')
    _float_array(selected_probabilities, 'captured selected probabilities', 2)
    _require(selected_probabilities.shape == (t, k) and selected_probabilities.dtype == np.float32,
             'Captured probability slot shape/dtype differs')
    routing = np.zeros((t, num_experts), dtype=bool)
    probabilities = np.zeros((t, num_experts), dtype=np.float32)
    rows = np.arange(t)[:, None]
    routing[rows, selected_experts] = True
    probabilities[rows, selected_experts] = selected_probabilities
    return routing, probabilities


def topk_stability(reference_scores, absolute_bound, top_k, domain=None):
    """Certify unique top-k set only if every selected lower > excluded upper.

    The reference set is computed independently. Exact ties are not assigned a
    stable expert identity. The bound must be external or analytically derived;
    this function never estimates it from candidate observations.
    """
    _float_array(reference_scores, 'reference scores', 2)
    tokens, experts = reference_scores.shape
    _require(type(top_k) is int and 0 < top_k <= experts, 'Invalid top_k')
    bound = np.asarray(absolute_bound, dtype=np.float64)
    _require(bound.shape == reference_scores.shape and np.isfinite(bound).all()
             and (bound >= 0).all(), 'Invalid stability envelope')
    domain = np.ones(reference_scores.shape, dtype=bool) if domain is None else domain
    _require(isinstance(domain, np.ndarray) and domain.dtype == bool
             and domain.shape == reference_scores.shape, 'Invalid stability domain mask')
    if top_k == experts:
        reference_set = np.ones_like(domain)
        stable = domain.all(axis=1)
        margins = [None] * tokens
    else:
        cut = np.partition(reference_scores, experts - top_k, axis=1)[:, -top_k]
        reference_set = reference_scores >= cut[:, None]
        unique = reference_set.sum(axis=1) == top_k
        # Outward endpoints: ordinary FP64 +/- is not claimed exact.
        lower = np.nextafter(reference_scores - bound, -np.inf)
        upper = np.nextafter(reference_scores + bound, np.inf)
        selected_lower = np.where(reference_set, lower, np.inf).min(axis=1)
        excluded_upper = np.where(~reference_set, upper, -np.inf).max(axis=1)
        gap = selected_lower - excluded_upper
        stable = unique & domain.all(axis=1) & (selected_lower > excluded_upper)
        margins = [float(g) if np.isfinite(g) else None for g in gap]
    return dict(stable=stable, reference_set=reference_set,
        report=_result('pass' if stable.all() else 'inconclusive', 'strict_interval_separation',
            tokens=tokens, stable_tokens=int(stable.sum()), unstable_tokens=int((~stable).sum()),
            stable_token_mask=stable.tolist(), conservative_cut_margins=margins,
            guarantee='unique set only for scores within the declared intervals'))


def _validate_endpoint(endpoint):
    _require(isinstance(endpoint.label, str) and bool(endpoint.label), 'Missing endpoint label')
    x, w, z = endpoint.inputs, endpoint.weights, endpoint.logits
    _float_array(x, 'inputs', 2); _float_array(w, 'weights', 2); _float_array(z, 'logits', 2)
    tokens, width = x.shape; experts = w.shape[0]
    _require(w.shape[1] == width and z.shape == (tokens, experts), 'Endpoint projection shapes differ')
    _require(isinstance(endpoint.expert_ids, tuple) and len(endpoint.expert_ids) == experts
             and all(type(i) is int for i in endpoint.expert_ids)
             and len(set(endpoint.expert_ids)) == experts, 'Missing/duplicate canonical expert identities')
    _require(isinstance(endpoint.coordinates, tuple) and len(endpoint.coordinates) == tokens
             and all(isinstance(c, tuple) and c and all(type(i) is int for i in c)
                     for c in endpoint.coordinates)
             and len(set(endpoint.coordinates)) == tokens, 'Missing/duplicate/noninteger token coordinates')
    k = endpoint.contract.top_k
    _require(type(k) is int and 0 < k <= experts, 'Invalid top_k')
    routing = endpoint.routing_map
    _require(isinstance(routing, np.ndarray) and routing.dtype == bool and routing.shape == z.shape,
             'Routing map must be an observed boolean token/expert matrix')
    _float_array(endpoint.probabilities, 'routing probabilities', 2)
    _require(endpoint.probabilities.dtype == np.float32 and endpoint.probabilities.shape == z.shape,
             'Routing probabilities must be sparse FP32 token/expert values')
    return tokens, experts, k


def _selection(endpoint, score_bound):
    contract = endpoint.contract
    if contract.selection_domain == 'logits':
        return endpoint.logits, None
    if contract.selection_domain != 'full_softmax':
        return None, _result('inconclusive', 'unsupported_selection_domain')
    if endpoint.selection_scores is None:
        return None, _result('inconclusive', 'actual_full_softmax_selection_scores_not_captured')
    scores = _float_array(endpoint.selection_scores, 'actual full softmax scores', 2)
    _require(scores.shape == endpoint.logits.shape and scores.dtype == np.float32,
             'Selection score shape/dtype differs')
    if ((scores < 0) | (scores > 1)).any():
        return scores, _result('fail', 'full_softmax_scores_outside_unit_interval')
    check = _declared_comparison(softmax64(endpoint.logits), scores, score_bound,
                                'full_softmax_error_bound_not_declared')
    return scores, check


def audit_endpoint(endpoint, *, probability_bound=None, probability_sum_bound=None,
                   selection_score_bound=None):
    """Check own-operand projection, own exact top-k and own normalization."""
    tokens, experts, k = _validate_endpoint(endpoint)
    c = endpoint.contract; checks = {}
    dtypes_supported = (_effective_dtype(endpoint.inputs, c.input_dtype, 'inputs')
        & _effective_dtype(endpoint.weights, c.weight_dtype, 'weights')
        & _effective_dtype(endpoint.logits, c.logit_dtype, 'logits'))
    model_supported = (dtypes_supported and c.input_dtype == c.weight_dtype == 'bfloat16'
                       and c.logit_dtype == 'float32'
                       and c.projection_arithmetic == 'fp32_mul_add_or_fma')
    same_input = _result('inconclusive', 'projection_arithmetic_or_dtype_not_supported')
    if model_supported:
        p = fp32_projection(endpoint.inputs, endpoint.weights)
        actual_normal = ((endpoint.logits == 0) | (np.abs(endpoint.logits) >= np.finfo(np.float32).tiny))
        domain = p['domain'] & actual_normal
        error = np.abs(p['reference'] - endpoint.logits.astype(np.float64))
        bad = domain & (error > p['absolute_bound'])
        checks['projection'] = _result('fail' if bad.any() else 'pass' if domain.all() else 'inconclusive',
            'analytic_same_operand_fp32_dot_bound', compared_elements=int(domain.sum()),
            excluded_elements=int((~domain).sum()), violating_elements=int(bad.sum()),
            max_abs=float(error.max()), max_bound=float(p['absolute_bound'].max()),
            domain_exclusions={**p['domain_exclusions'], 'observed_subnormal_logit': int((~actual_normal).sum())},
            assumptions=p['assumptions'])
        if c.selection_domain == 'logits':
            same_input = topk_stability(p['reference'], p['absolute_bound'], k, domain)['report']
            same_input['scope'] = 'same operands only; no cross-end activation/weight perturbation guarantee'
        else:
            same_input = _result('inconclusive', 'rounded_full_softmax_stability_needs_a_separate_score_envelope')
    else:
        checks['projection'] = _result('inconclusive', 'projection_arithmetic_or_dtype_not_supported')
    scores, score_check = _selection(endpoint, selection_score_bound)
    if score_check is not None:
        checks['selection_scores'] = score_check
    checks['topk'] = (legal_topk(scores, endpoint.routing_map, k) if scores is not None
                      else _result('inconclusive', 'actual_selection_scores_unavailable'))
    probs, route = endpoint.probabilities, endpoint.routing_map
    malformed_probs = (probs < 0) | (probs > 1) | ((~route) & (probs != 0))
    checks['probability_support'] = _result('fail' if malformed_probs.any() else 'pass',
        'nonnegative_unit_interval_zero_outside_actual_routes',
        violating_elements=int(malformed_probs.sum()))
    supported_probability = ((c.selection_domain == 'logits' and c.probability_rule == 'selected_softmax')
        or (c.selection_domain == 'full_softmax' and c.probability_rule == 'full_softmax_selected_renorm'))
    if not supported_probability:
        checks['probabilities'] = _result('inconclusive', 'unsupported_probability_rule')
    elif not (route.sum(axis=1) == k).all():
        checks['probabilities'] = _result('fail', 'route_count_invalid_for_normalization')
    else:
        expected = np.zeros((tokens, experts), dtype=np.float64)
        if c.probability_rule == 'selected_softmax':
            selected_logits = endpoint.logits[route].reshape(tokens, k)
            expected[route] = softmax64(selected_logits).reshape(-1)
        else:
            full = softmax64(endpoint.logits)
            selected = np.where(route, full, 0.)
            denominator = selected.sum(axis=1, keepdims=True)
            if (denominator == 0).any():
                checks['probabilities'] = _result('fail', 'selected_full_softmax_mass_is_zero')
            else:
                expected = selected / denominator
        if 'probabilities' not in checks:
            checks['probabilities'] = _declared_comparison(expected, probs, probability_bound,
                                                         'softmax_probability_error_bound_not_declared')
    sums = probs.astype(np.float64).sum(axis=1)
    checks['probability_sum'] = _declared_comparison(np.ones(tokens), sums, probability_sum_bound,
                                                   'probability_sum_error_bound_not_declared')
    return dict(schema_version=1, scope='local_routing_semantics', label=endpoint.label,
        status=_aggregate(checks), checks=checks, same_input_stability=same_input,
        training_acceptance=False, native_routes_used_as_reference=False)


def audit_pair(reference, candidate, *, cross_end_bound=None):
    """Same-input stability is intentionally not a cross-end perturbation budget.

    The external budget covers candidate actual selection logits relative to
    the independently recomputed reference projection. It must include upstream
    activation/weight drift and projection error. Ref-own projection error is
    independently analytic. Without such a budget even identical routes remain
    inconclusive. This diagnoses a declared envelope; it does not qualify it.
    """
    rt, re, rk = _validate_endpoint(reference); ct, ce, ck = _validate_endpoint(candidate)
    _require((rt, re, rk) == (ct, ce, ck) and reference.coordinates == candidate.coordinates
             and reference.expert_ids == candidate.expert_ids,
             'Cross-end token/expert/top_k identities differ')
    _require(reference.inputs.shape == candidate.inputs.shape
             and reference.weights.shape == candidate.weights.shape, 'Cross-end parameter dimensions differ')
    mismatches = (reference.routing_map != candidate.routing_map).any(axis=1)
    result = dict(schema_version=1, scope='cross_end_route_stability_diagnostic',
        route_set_mismatch_tokens=int(mismatches.sum()), training_acceptance=False,
        native_routes_used_as_reference=False, budget_estimated_from_observed_difference=False)
    checks = {}
    for side, endpoint in (('reference', reference), ('candidate', candidate)):
        local = audit_endpoint(endpoint)
        for name in ('projection', 'topk', 'selection_scores'):
            if name in local['checks']:
                checks[side + '_' + name] = local['checks'][name]
    if cross_end_bound is None:
        checks['cross_end_stability'] = _result('inconclusive', 'independent_upstream_cross_end_budget_not_declared')
        return dict(result, status=_aggregate(checks), checks=checks)
    wanted = dict(input_dtype='bfloat16', weight_dtype='bfloat16', logit_dtype='float32',
                  projection_arithmetic='fp32_mul_add_or_fma', selection_domain='logits')
    if any(getattr(endpoint.contract, key) != value
           for endpoint in (reference, candidate) for key, value in wanted.items()):
        checks['cross_end_stability'] = _result('inconclusive', 'cross_end_score_domain_not_supported')
        return dict(result, status=_aggregate(checks), checks=checks)
    projection = fp32_projection(reference.inputs, reference.weights)
    ref_values = projection['reference']; ext = cross_end_bound.array(ref_values.shape)
    # No candidate route participates in construction of reference_set.
    envelope = np.maximum(projection['absolute_bound'], ext)
    stability = topk_stability(ref_values, envelope, rk, projection['domain'])
    checks['cross_end_budget'] = _comparison(ref_values, candidate.logits, ext, cross_end_bound.source)
    stable = stability['stable']; expected = stability['reference_set']
    unexpected = stable & (((reference.routing_map != expected).any(axis=1))
                           | ((candidate.routing_map != expected).any(axis=1)))
    checks['stable_sets'] = _result('fail' if unexpected.any() else 'pass' if stable.all() else 'inconclusive',
        'independently_computed_reference_set_with_strict_interval_separation',
        stable_tokens=int(stable.sum()), unstable_tokens=int((~stable).sum()),
        violating_stable_tokens=int(unexpected.sum()))
    return dict(result, status=_aggregate(checks), checks=checks, stability=stability['report'],
                bound_source=cross_end_bound.source)


def _same_values(expected, observed, name):
    _require(isinstance(observed, np.ndarray) and expected.shape == observed.shape
             and expected.dtype == observed.dtype, name + ' shape/storage dtype differs')
    _float_array(observed, name)
    different = (np.ascontiguousarray(expected).view(np.uint8).reshape(expected.size, expected.itemsize)
                 != np.ascontiguousarray(observed).view(np.uint8).reshape(observed.size, observed.itemsize)).any(axis=1)
    return _result('fail' if different.any() else 'pass', 'exact_declared_value_transport_including_signed_zero',
                   elements=int(expected.size), different_elements=int(different.sum()))


def audit_dispatch(endpoint, evidence):
    """Validate actual expanded token/expert permutation and inverse scatter."""
    tokens, experts, k = _validate_endpoint(endpoint)
    if evidence is None:
        return dict(scope='complete_logical_dispatch', status='inconclusive', training_acceptance=False,
                    reason='actual_dispatch_not_captured')
    c = evidence.contract
    if c.domain != 'complete_no_drop_no_pad' or c.order not in ('expert_token', 'expert_grouped_any'):
        return dict(scope='complete_logical_dispatch', status='inconclusive', training_acceptance=False,
                    reason='dispatch_domain_or_order_not_supported')
    n = tokens * k
    row, expert = evidence.row_token, evidence.row_expert
    _integer_array(row, 'dispatch row_token', (n,)); _integer_array(expert, 'dispatch row_expert', (n,))
    _integer_array(evidence.expert_counts, 'actual expert_counts', (experts,))
    _require(((0 <= row) & (row < tokens)).all() and ((0 <= expert) & (expert < experts)).all(),
             'Dispatch token/expert coordinate out of range')
    pair_ids = row.astype(np.int64) * experts + expert.astype(np.int64)
    expected_pairs = np.flatnonzero(endpoint.routing_map.reshape(-1))
    complete = len(expected_pairs) == n and np.array_equal(np.sort(pair_ids), expected_pairs)
    grouped = bool(np.all(expert[1:] >= expert[:-1]))
    order_ok = grouped and (c.order != 'expert_token'
        or bool(np.all((expert[1:] != expert[:-1]) | (row[1:] > row[:-1]))))
    counts = np.bincount(expert.astype(np.int64), minlength=experts)
    checks = dict(pair_permutation=_result('pass' if complete else 'fail',
                    'bijection_over_observed_selected_token_expert_pairs', rows=n,
                    distinct_pairs=int(len(np.unique(pair_ids)))),
        expert_order=_result('pass' if order_ok else 'fail', 'declared_expert_row_order'),
        expert_counts=_result('pass' if np.array_equal(counts, evidence.expert_counts) else 'fail',
                             'counts_recomputed_from_actual_rows'))
    if evidence.combine_token is None:
        checks['inverse'] = _result('inconclusive', 'actual_inverse_scatter_indices_not_captured')
    else:
        inverse = _integer_array(evidence.combine_token, 'actual inverse token indices', (n,))
        checks['inverse'] = _result('pass' if np.array_equal(inverse, row) else 'fail',
                                   'actual_expanded_row_to_original_token_map')
    supported = (_effective_dtype(endpoint.inputs, c.input_dtype, 'dispatch source inputs')
                 and c.input_dtype == endpoint.contract.input_dtype)
    if supported:
        checks['inputs'] = _same_values(endpoint.inputs[row], evidence.dispatched_inputs, 'dispatched inputs')
    else:
        checks['inputs'] = _result('inconclusive', 'dispatch_input_dtype_not_supported')
    if c.probability_dtype in ('float32', 'bfloat16'):
        expected_probs = _round(endpoint.probabilities[row, expert], c.probability_dtype)
        checks['probabilities'] = _same_values(expected_probs, evidence.dispatched_probabilities,
                                              'dispatched probabilities')
    else:
        checks['probabilities'] = _result('inconclusive', 'dispatch_probability_dtype_not_supported')
    return dict(scope='complete_logical_dispatch', status=_aggregate(checks), checks=checks,
        training_acceptance=False, distributed_ownership_or_replica_cover_verified=False)


def audit_combine(endpoint, dispatch, evidence, *, output_bound=None):
    """Check declared row weighting and accumulation using actual row outputs.

    For preweighted expert outputs, this verifies only inverse transport/sum;
    probability multiplication before the captured output remains inconclusive.
    No unweighted counterfactual expert outputs are manufactured.
    """
    tokens, _, k = _validate_endpoint(endpoint)
    if dispatch is None or evidence is None:
        return dict(scope='weighted_combine', status='inconclusive', training_acceptance=False,
                    reason='actual_dispatch_or_expert_output_or_combine_output_not_captured')
    transport = audit_dispatch(endpoint, dispatch)
    checks = dict(dispatch=transport)
    c = evidence.contract
    dtypes = ('bfloat16', 'float32', 'float64')
    supported = (bool(isinstance(c.source, str) and c.source.strip())
        and c.weighting in ('post_expert', 'preweighted')
        and c.expert_output_dtype in dtypes and c.accumulation_dtype in dtypes and c.output_dtype in dtypes
        and (c.product_dtype in dtypes if c.weighting == 'post_expert' else c.product_dtype == 'already_weighted')
        and (c.accumulation_order == 'row_order' or (c.accumulation_order == 'two_routes' and k == 2)))
    if not supported:
        checks['combine'] = _result('inconclusive', 'combine_arithmetic_boundary_or_order_not_declared')
    elif transport['status'] != 'pass':
        checks['combine'] = _result('inconclusive', 'dispatch_not_established')
    else:
        values = _float_array(evidence.expert_outputs, 'actual expert outputs', 2)
        actual = _float_array(evidence.combined_output, 'actual combine output', 2)
        _require(values.shape[0] == tokens * k and actual.shape == (tokens, values.shape[1]),
                 'Combine output dimensions differ')
        _effective_dtype(values, c.expert_output_dtype, 'actual expert outputs')
        _effective_dtype(actual, c.output_dtype, 'actual combine output')
        if c.weighting == 'post_expert':
            # Compare against each endpoint's own verified routed probability.
            products = values.astype(np.float64) * dispatch.dispatched_probabilities.astype(np.float64)[:, None]
            contributions = _round(products, c.product_dtype)
        else:
            contributions = values
            checks['probability_weighting'] = _result('inconclusive',
                'preweighted_output_does_not_observe_internal_probability_multiplication')
        output = np.zeros(actual.shape, dtype=np.float64)
        for index, token in enumerate(dispatch.combine_token):
            output[token] = _round(output[token] + contributions[index].astype(np.float64), c.accumulation_dtype)
        expected = _round(output, c.output_dtype)
        checks['combine'] = _declared_comparison(expected, actual, output_bound,
                                                 'combine_output_error_bound_not_declared')
    return dict(scope='weighted_combine', status=_aggregate(checks), checks=checks,
        training_acceptance=False, arithmetic_contract_source=c.source,
        expert_function_correctness_verified=False)


def audit_layer(endpoint, *, dispatch=None, combine=None, probability_bound=None,
                probability_sum_bound=None, selection_score_bound=None, combine_output_bound=None):
    """Full local catalog: missing dispatch/combine cannot disappear from status."""
    checks = dict(routing=audit_endpoint(endpoint, probability_bound=probability_bound,
        probability_sum_bound=probability_sum_bound, selection_score_bound=selection_score_bound),
        dispatch=audit_dispatch(endpoint, dispatch),
        combine=audit_combine(endpoint, dispatch, combine, output_bound=combine_output_bound))
    return dict(schema_version=1, scope='local_moe_semantic_checks', status=_aggregate(checks), checks=checks,
                training_acceptance=False, distributed_matrix_qualification=False)
