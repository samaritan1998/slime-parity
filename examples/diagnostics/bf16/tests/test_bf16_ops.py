"""Independent FP64 local-function checks of explicit BF16 arithmetic boundaries."""
import pytest
import torch

import bf16_ops as ops


def _leaf(value):
    return value.detach().bfloat16().requires_grad_()


def _round_like(reference, actual):
    expected = reference.bfloat16().to(actual.dtype)
    # The test uses random, nonsaturating values away from BF16 ties. Tight
    # closeness to the rounded FP64 expectation bounds FP32 arithmetic error.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(5, 7), (2, 3, 7)])
def test_linear_forward_inputgrad_fp64_and_unrounded_weightgrad(shape):
    torch.manual_seed(129)
    x = _leaf(torch.randn(shape))
    w = torch.randn(11, 7, requires_grad=True)
    upstream = torch.randn(*shape[:-1], 11).bfloat16()
    y = ops.linear(x, w)
    y.backward(upstream)
    xd, wd, gd = x.detach().double(), w.detach().bfloat16().double(), upstream.double()
    expected = torch.einsum('...i,oi->...o', xd, wd)
    _round_like(expected, y)
    _round_like(torch.einsum('...o,oi->...i', gd, wd), x.grad)
    expected_wgrad = torch.einsum('mo,mi->oi', gd.reshape(-1, 11), xd.reshape(-1, 7))
    assert w.grad.dtype == torch.float32
    torch.testing.assert_close(w.grad.double(), expected_wgrad, atol=3e-6, rtol=3e-6)
    assert not torch.equal(w.grad, w.grad.bfloat16().float())


def test_linear_microbatch_gradients_accumulate_in_fp32_and_use_updated_master():
    torch.manual_seed(81)
    w = torch.randn(8, 5, requires_grad=True)
    expected = torch.zeros_like(w, dtype=torch.float64)
    for count in (2, 3, 1):
        x = torch.randn(count, 5).bfloat16()
        grad = torch.randn(count, 8).bfloat16()
        ops.linear(x, w).backward(grad)
        expected += grad.double().T @ x.double()
    torch.testing.assert_close(w.grad.double(), expected, atol=2e-6, rtol=2e-6)
    old = w.detach().clone()
    with torch.no_grad():
        w.add_(-.01 * w.grad)
    x = torch.ones(2, 5, dtype=torch.bfloat16)
    y = ops.linear(x, w)
    _round_like(x.double() @ w.detach().bfloat16().double().T, y)
    assert not torch.equal(y, (x.double() @ old.bfloat16().double().T).bfloat16())


@pytest.mark.parametrize('writeback', ['bfloat16', 'float32'])
def test_embedding_repeated_indices_padding_and_gradient_writeback(writeback):
    torch.manual_seed(72)
    w = torch.randn(9, 5, requires_grad=True)
    indices = torch.tensor([[1, 2, 1, 0], [2, 1, 8, 0]])
    grad = torch.randn(2, 4, 5).bfloat16()
    y = ops.embedding(indices, w, padding_idx=0, gradient_writeback=writeback)
    y.backward(grad)
    _round_like(w.detach().double()[indices], y)
    # Independent one-hot contraction; excludes padding from the derivative.
    onehot = torch.nn.functional.one_hot(indices, 9).double()
    onehot[..., 0] = 0
    expected = torch.einsum('blv,bld->vd', onehot, grad.double())
    assert w.grad.dtype == torch.float32
    if writeback == 'bfloat16':
        _round_like(expected, w.grad)
    else:
        torch.testing.assert_close(w.grad.double(), expected, atol=0, rtol=0)
        assert not torch.equal(w.grad, w.grad.bfloat16().float())
    assert torch.count_nonzero(w.grad[0]) == 0


def test_embedding_quantizes_each_microbatch_gradient_before_fp32_accumulation():
    torch.manual_seed(834)
    weight = torch.randn(3, 5, requires_grad=True)
    indices = torch.tensor([1, 1, 1])
    rounded_micros, exact_micros = [], []
    for _ in range(3):
        gradient = torch.randn(3, 5).bfloat16()
        ops.embedding(indices, weight).backward(gradient)
        exact = gradient.double().sum(0)
        exact_micros.append(exact)
        rounded_micros.append(exact.bfloat16().float())
    expected = sum(rounded_micros)
    assert torch.equal(weight.grad[1], expected)
    assert not torch.equal(expected, sum(exact_micros).bfloat16().float())
    assert not torch.equal(expected, sum(exact_micros).float())


def test_embedding_chunk32_writeback_is_not_one_global_rounding():
    torch.manual_seed(717)
    weight = torch.randn(3, 7, requires_grad=True)
    ids = torch.ones(65, dtype=torch.long)
    ids[::5] = 0  # Padding keeps its token positions in the 32-token chunks.
    gradient = torch.randn(65, 7).bfloat16()
    ops.embedding(ids, weight, padding_idx=0).backward(gradient)
    expected = torch.zeros(7, dtype=torch.bfloat16)
    for first in (0, 32, 64):
        active = ids[first:first + 32] == 1
        partial = gradient[first:first + 32][active].double().sum(0).bfloat16()
        expected = (expected.double() + partial.double()).bfloat16()
    assert torch.equal(weight.grad[1], expected.float())
    wrong = gradient[ids == 1].double().sum(0).bfloat16().float()
    assert not torch.equal(weight.grad[1], wrong)
    assert torch.count_nonzero(weight.grad[0]) == 0


def test_rmsnorm_fp64_jacobian_and_bf16_gamma_gradient_boundary():
    torch.manual_seed(24)
    x = _leaf(torch.randn(2, 3, 7))
    gamma = torch.randn(7, requires_grad=True)
    grad = torch.randn_like(x)
    y = ops.rmsnorm(x, gamma, eps=1e-6)
    y.backward(grad)
    xd = x.detach().double()
    gd = gamma.detach().bfloat16().double()
    denom = (torch.sum(xd * xd, dim=-1, keepdim=True) / 7 + 1e-6).sqrt()
    expected = xd * gd / denom
    _round_like(expected, y)
    # Closed-form Jacobian product independent of autograd through the FP32 graph.
    incoming = grad.double() * gd
    dx = incoming / denom - xd * (incoming * xd).sum(-1, keepdim=True) / (7 * denom.pow(3))
    dg = (grad.double() * xd / denom).sum((0, 1))
    _round_like(dx, x.grad)
    _round_like(dg, gamma.grad)
    assert torch.equal(gamma.grad, gamma.grad.bfloat16().float())


def test_rope_default_positions_fp64_rotation_and_transpose_gradient():
    torch.manual_seed(95)
    x = _leaf(torch.randn(2, 3, 5, 8))
    ids = torch.tensor([[0, 1, 7, 31, 64], [0, 2, 8, 32, 63]])
    inverse = 1 / (1_000_000 ** (torch.arange(0, 8, 2).float() / 8))
    cos, sin = ops.rope_coefficients(ids, inverse)
    angles = ids.double()[..., None] * inverse.double()
    torch.testing.assert_close(cos[..., :4].double(), angles.cos(), atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(sin[..., :4].double(), angles.sin(), atol=2e-7, rtol=2e-7)
    grad = torch.randn_like(x)
    y = ops.rope(x, cos, sin)
    y.backward(grad)
    first, second = x.detach().double().chunk(2, -1)
    c, s = cos[:, None, :, :4].double(), sin[:, None, :, :4].double()
    expected = torch.cat((c * first - s * second, s * first + c * second), -1)
    _round_like(expected, y)
    assert torch.equal(y[:, :, 0], x[:, :, 0])
    u, v = grad.double().chunk(2, -1)
    _round_like(torch.cat((c * u + s * v, -s * u + c * v), -1), x.grad)


def test_swiglu_fp64_forward_and_analytic_backward():
    torch.manual_seed(213)
    gate, up = _leaf(torch.randn(2, 3, 17)), _leaf(torch.randn(2, 3, 17))
    grad = torch.randn_like(gate)
    y = ops.swiglu(gate, up)
    y.backward(grad)
    g, u = gate.detach().double(), up.detach().double()
    sigmoid = 1 / (1 + torch.exp(-g))
    _round_like(g * sigmoid * u, y)
    _round_like(grad.double() * u * (sigmoid + g * sigmoid * (1 - sigmoid)), gate.grad)
    _round_like(grad.double() * g * sigmoid, up.grad)


@pytest.mark.parametrize('causal', [False, True])
def test_attention_fp64_gqa_forward_and_separate_backward(causal):
    torch.manual_seed(619)
    query = _leaf(torch.randn(2, 4, 5, 8))
    key, value = _leaf(torch.randn(2, 2, 5, 8)), _leaf(torch.randn(2, 2, 5, 8))
    grad = torch.randn_like(query)
    y = ops.attention_fp32(query, key, value, causal=causal)
    y.backward(grad)
    q, k, v = (item.detach().double() for item in (query, key, value))
    k = k[:, :, None].expand(2, 2, 2, 5, 8).reshape(2, 4, 5, 8)
    v = v[:, :, None].expand(2, 2, 2, 5, 8).reshape(2, 4, 5, 8)
    score = torch.einsum('bhid,bhjd->bhij', q, k) / 8 ** .5
    if causal:
        mask = torch.arange(5)[None, :] > torch.arange(5)[:, None]
        score = score.masked_fill(mask, -torch.inf)
    # Explicit normalized exponentials, then independent analytic softmax VJP.
    exponent = (score - score.max(-1, keepdim=True).values).exp()
    probability = exponent / exponent.sum(-1, keepdim=True)
    expected = torch.einsum('bhij,bhjd->bhid', probability, v)
    _round_like(expected, y)
    dy = grad.double()
    dp = torch.einsum('bhid,bhjd->bhij', dy, v)
    ds = probability * (dp - (dp * probability).sum(-1, keepdim=True)) / 8 ** .5
    dq = torch.einsum('bhij,bhjd->bhid', ds, k)
    dk = torch.einsum('bhij,bhid->bhjd', ds, q).reshape(2, 2, 2, 5, 8).sum(2)
    dv = torch.einsum('bhij,bhid->bhjd', probability, dy).reshape(2, 2, 2, 5, 8).sum(2)
    for actual, expected in ((query.grad, dq), (key.grad, dk), (value.grad, dv)):
        _round_like(expected, actual)
    if causal:
        assert torch.equal(y[:, :, 0], value.detach().repeat_interleave(2, dim=1)[:, :, 0])


def test_outer_autocast_does_not_change_explicit_contract():
    torch.manual_seed(133)
    x = torch.randn(3, 7).bfloat16()
    w = torch.randn(11, 7, requires_grad=True)
    expected = ops.linear(x, w)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = ops.linear(x, w)
    assert torch.equal(actual, expected)


def test_invalid_storage_and_unsupported_options_fail():
    with pytest.raises(ValueError, match='BF16'):
        ops.linear(torch.ones(2, 3), torch.ones(4, 3))
    with pytest.raises(ValueError, match='FP32 master'):
        ops.linear(torch.ones(2, 3).bfloat16(), torch.ones(4, 3).bfloat16())
    with pytest.raises(ValueError, match='writeback'):
        ops.embedding(torch.tensor([1]), torch.ones(3, 4), gradient_writeback='automatic')
    with pytest.raises(ValueError, match='padding index'):
        ops.embedding(torch.tensor([1]), torch.ones(3, 4), padding_idx=3)
    with pytest.raises(ValueError, match='epsilon'):
        ops.rmsnorm(torch.ones(2, 4).bfloat16(), torch.ones(4), eps=0)
