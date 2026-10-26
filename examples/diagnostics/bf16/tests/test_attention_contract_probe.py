"""Bounded CPU formula checks; no native runtime or GPU validation."""
import pytest
import torch

from attention_contract_probe import VARIANTS, evaluate_attention, fp64_mathematical_reference


def inputs(length=7, heads=4, kv_heads=2, seed=941):
    torch.manual_seed(seed)
    return [torch.randn(shape).bfloat16() for shape in
            ((length, heads, 32), (length, kv_heads, 32),
             (length, kv_heads, 32), (length, heads, 32))]


def test_fp64_closed_form_matches_separate_autograd_graph():
    q, k, v, do = inputs()
    actual = fp64_mathematical_reference(q, k, v, do)
    qd, kd, vd = [x.double().detach().requires_grad_() for x in (q, k, v)]
    score = torch.einsum('ihd,jhd->hij', qd, kd.repeat_interleave(2, 1)) / 32**.5
    score = score.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
    out = torch.einsum('hij,jhd->ihd', score.softmax(-1), vd.repeat_interleave(2, 1))
    out.backward(do.double())
    for role, expected in zip(('forward', 'q_gradient', 'k_gradient', 'v_gradient'),
                              (out, qd.grad, kd.grad, vd.grad)):
        torch.testing.assert_close(actual[role], expected, atol=2e-15, rtol=3e-13)


def test_explicit_gqa_grouping_matches_expanded_mha_then_bf16_sum():
    q, k, v, do = inputs(length=11, heads=6, kv_heads=2)
    grouped = evaluate_attention(q, k, v, do)
    expanded = evaluate_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), do)
    assert torch.equal(grouped['forward'], expanded['forward'])
    assert torch.equal(grouped['q_gradient'], expanded['q_gradient'])
    for role in ('k_gradient', 'v_gradient'):
        expected = expanded[role].float().reshape(11, 2, 3, 32).sum(2).bfloat16()
        assert torch.equal(grouped[role], expected)


def test_causal_prefix_independent_of_future_keys_values():
    q, k, v, do = inputs(length=13)
    a = evaluate_attention(q, k, v, do)
    altered_k, altered_v = k.clone(), v.clone()
    altered_k[9:] = 7
    altered_v[9:] = -9
    b = evaluate_attention(q, altered_k, altered_v, do)
    assert torch.equal(a['forward'][:9], b['forward'][:9])
    assert torch.equal(a['q_gradient'][:9], b['q_gradient'][:9])


def test_zero_upstream_and_output_dtypes_all_variants():
    q, k, v, do = inputs()
    for variant in VARIANTS:
        result = evaluate_attention(q, k, v, do.zero_(), variant=variant)
        assert all(x.dtype == torch.bfloat16 and torch.isfinite(x).all() for x in result.values())
        for role in ('q_gradient', 'k_gradient', 'v_gradient'):
            assert torch.count_nonzero(result[role]) == 0


def test_known_boundary_changes_are_observable():
    q, k, v, do = inputs(length=64)
    results = {name: evaluate_attention(q, k, v, do, variant=name) for name in VARIANTS}
    assert not torch.equal(results['math_fp32']['forward'], results['forward_exp_bf16']['forward'])
    assert not torch.equal(results['recompute_and_output_delta']['v_gradient'], results['backward_p_bf16']['v_gradient'])
    assert not torch.equal(results['backward_p_bf16']['q_gradient'], results['backward_p_ds_bf16']['q_gradient'])
    assert not torch.equal(results['backward_p_ds_bf16']['k_gradient'], results['fa2_contract']['k_gradient'])
    # Delta recomputation changes q/k derivatives, but not dV before P casting.
    assert not torch.equal(results['forward_exp_bf16']['q_gradient'], results['recompute_and_output_delta']['q_gradient'])


@pytest.mark.parametrize('length', [0, 129])
def test_multi_tile_or_empty_lengths_rejected(length):
    with pytest.raises(ValueError):
        evaluate_attention(*inputs(length=length))


def test_non_bf16_rejected():
    q, k, v, do = inputs()
    with pytest.raises(ValueError):
        evaluate_attention(q.float(), k, v, do)
