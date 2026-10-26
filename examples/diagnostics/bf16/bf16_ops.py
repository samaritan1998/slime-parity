"""Independent first-order BF16 arithmetic hypotheses; never an acceptance policy.

Only PyTorch arithmetic is used. FP32 Parameter storage retains optimizer-master
precision; effective forward parameters are BF16. Casting is treated as an
identity derivative with explicitly rounded activation/parameter gradients.
The finite-precision forward itself is not differentiable by finite differences.

This module does not import Slime, Megatron, TE or native captured outputs. CPU
tests compare each operator against a separately evaluated FP64 local function.
CUDA GEMM/reduction order and fused arithmetic still require separate evidence.
"""
import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


CONTRACTS = {
    'linear': {
        'forward': 'BF16 effective input and weight; BF16 GEMM output',
        'backward': 'BF16 output gradient; BF16 input gradient; FP32 weight-gradient GEMM result',
        'supported': 'dense bias-free linear; first-order gradients; unique FP32 master Parameter',
        'limits': 'Different GEMM tiling/reduction/FMA order is not reproduced. FP32 wgrad must not use TF32.',
    },
    'embedding': {
        'forward': 'Gather BF16-rounded effective weights',
        'backward': 'For each 32-token chunk, sum repeated-token gradients in token order in FP32; cast each partial to BF16 and add to BF16 weight grad; finally promote to FP32 master grad',
        'supported': 'NVIDIA PyTorch 2.11 small embedding path, at most 3072 indices; dense gradients; optional padding index; no frequency scaling',
        'limits': 'Conditional source contract, pending runtime kernel verification. Chunk boundaries retain padding positions. Direct FP32 scatter/writeback is an explicitly different diagnostic.',
    },
    'rmsnorm': {
        'forward': 'BF16 effective x/gamma; FP32 square/mean/rsqrt/scale; BF16 output',
        'backward': 'FP32 local derivative; BF16 x and gamma gradient writeback; gamma promoted to FP32 master grad',
        'supported': 'last-dimension RMS normalization; zero_centered_gamma=False',
        'limits': 'Matches documented dtype boundaries, not TE reduction/FMA ordering or exact kernel arithmetic.',
    },
    'rope': {
        'forward': 'FP32 coefficients and split-half multiply/add; BF16 output',
        'backward': 'FP32 transpose rotation; BF16 input-gradient writeback',
        'supported': 'default split-half RoPE; explicit positions and FP32 inverse frequencies',
        'limits': 'No fused GPU trigonometric/FMA equivalence; no learned or dynamic RoPE.',
    },
    'swiglu': {
        'forward': 'FP32 SiLU and product on BF16 inputs; BF16 output',
        'backward': 'FP32 local derivative; BF16 gate/up gradient writeback',
        'supported': 'bias-free SwiGLU',
        'limits': 'Hypothesis only: generated Inductor/Triton arithmetic has not been verified.',
    },
    'attention_fp32': {
        'forward': 'BF16 Q/K/V promoted to FP32; FP32 scores/softmax/PV; final BF16 output',
        'backward': 'Ordinary FP32 graph; BF16 Q/K/V gradient writeback',
        'supported': 'BHLD, grouped-query attention, optional causal mask, no dropout/cache',
        'limits': 'Diagnostic mathematical reference, not a FlashAttention contract. Online softmax, probability casts, dP/dS precision, tile/reduction order and backend selection remain unresolved.',
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bf16(value, name):
    _require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16,
             f'{name} must be a BF16 tensor')


def _master(value, name):
    _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32,
             f'{name} must have FP32 master storage')


def _scope(device):
    # Explicit operations must not inherit an outer BF16 autocast context.
    return torch.autocast(device_type=device.type, enabled=False)


def _check_fp32_gemm(device):
    if device.type == 'cuda':
        _require(not torch.backends.cuda.matmul.allow_tf32,
                 'Disable TF32 before running explicit FP32 GEMM diagnostics')


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        _bf16(inputs, 'linear input')
        _master(weight, 'linear weight')
        _require(weight.ndim == 2 and inputs.ndim >= 2 and inputs.shape[-1] == weight.shape[1],
                 'linear dimensions differ')
        _require(inputs.device == weight.device, 'linear devices differ')
        _check_fp32_gemm(inputs.device)
        effective_weight = weight.to(torch.bfloat16)
        ctx.save_for_backward(inputs, effective_weight)
        with _scope(inputs.device):
            return F.linear(inputs, effective_weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        inputs, weight = ctx.saved_tensors
        _bf16(output_gradient, 'linear output gradient')
        _check_fp32_gemm(inputs.device)
        with _scope(inputs.device):
            gradient = output_gradient.reshape(-1, weight.shape[0])
            input_gradient = (gradient @ weight).reshape(inputs.shape) if ctx.needs_input_grad[0] else None
            weight_gradient = (gradient.float().T @ inputs.reshape(-1, weight.shape[1]).float()
                               if ctx.needs_input_grad[1] else None)
        return input_gradient, weight_gradient


def linear(inputs, weight):
    """BF16 linear with directly accumulated FP32 wgrad; no bias."""
    return _Linear.apply(inputs, weight)


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight, padding_idx, gradient_writeback):
        _master(weight, 'embedding weight')
        _require(weight.ndim == 2 and indices.dtype == torch.long and indices.device == weight.device,
                 'embedding expects colocated integer indices and matrix weights')
        _require(gradient_writeback in ('bfloat16', 'float32'), 'unknown embedding gradient writeback')
        _require(gradient_writeback != 'bfloat16' or indices.numel() <= 3072,
                 'BF16 embedding diagnostic only covers the small CUDA kernel (at most 3072 indices)')
        _require(padding_idx is None or type(padding_idx) is int and 0 <= padding_idx < weight.shape[0],
                 'padding index outside vocabulary')
        _require(bool(((indices >= 0) & (indices < weight.shape[0])).all()), 'index outside vocabulary')
        ctx.save_for_backward(indices)
        ctx.weight_shape, ctx.padding_idx, ctx.gradient_writeback = weight.shape, padding_idx, gradient_writeback
        with _scope(weight.device):
            return weight.to(torch.bfloat16)[indices]

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        (indices,) = ctx.saved_tensors
        _bf16(output_gradient, 'embedding output gradient')
        with _scope(output_gradient.device):
            flat_indices = indices.reshape(-1)
            gradients = output_gradient.reshape(-1, ctx.weight_shape[1]).float()
            if ctx.gradient_writeback == 'bfloat16':
                # PyTorch v2.11 Embedding.cu's NVIDIA feature kernel handles
                # <=3072 indices in ordered chunks of 32. Each chunk's first
                # matching token owns an FP32 partial, but writes/accumulates
                # it into BF16 grad_weight. Do not compact padding positions.
                weight_gradient = torch.zeros(ctx.weight_shape, dtype=torch.bfloat16, device=output_gradient.device)
                ids = flat_indices.detach().cpu().tolist()
                for first in range(0, len(ids), 32):
                    groups = {}
                    for index in range(first, min(first + 32, len(ids))):
                        token = ids[index]
                        if token != ctx.padding_idx:
                            groups.setdefault(token, []).append(index)
                    for token, positions in groups.items():
                        partial = gradients[positions[0]].clone()
                        for position in positions[1:]:
                            partial.add_(gradients[position])
                        weight_gradient[token].add_(partial.bfloat16())
                weight_gradient = weight_gradient.float()
            else:
                if ctx.padding_idx is not None:
                    keep = flat_indices != ctx.padding_idx
                    flat_indices, gradients = flat_indices[keep], gradients[keep]
                weight_gradient = torch.zeros(ctx.weight_shape, dtype=torch.float32, device=output_gradient.device)
                weight_gradient.index_add_(0, flat_indices, gradients)
        return None, weight_gradient, None, None


def embedding(indices, weight, padding_idx=None, *, gradient_writeback='bfloat16'):
    """BF16 gather; BF16 parameter-gradient boundary unless explicitly overridden."""
    return _Embedding.apply(indices, weight, padding_idx, gradient_writeback)


def rmsnorm(inputs, gamma, eps=1e-6):
    _bf16(inputs, 'RMSNorm input')
    _master(gamma, 'RMSNorm gamma')
    _require(gamma.ndim == 1 and gamma.shape[0] == inputs.shape[-1] and gamma.device == inputs.device,
             'RMSNorm gamma shape/device differs')
    _require(isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0, 'RMSNorm epsilon must be positive')
    with _scope(inputs.device):
        effective_gamma = gamma.bfloat16().float()
        fp32 = inputs.float()
        inverse_rms = torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)
        return (fp32 * inverse_rms * effective_gamma).bfloat16()


def rope_coefficients(position_ids, inverse_frequencies):
    _require(position_ids.ndim == 2 and position_ids.dtype == torch.long,
             'RoPE position IDs must be a batch-by-sequence integer matrix')
    _require(inverse_frequencies.ndim == 1 and inverse_frequencies.dtype == torch.float32
             and inverse_frequencies.device == position_ids.device and not inverse_frequencies.requires_grad,
             'RoPE inverse frequencies must be colocated constant FP32 values')
    _require(bool((position_ids >= 0).all()), 'RoPE positions must be nonnegative')
    with torch.no_grad(), _scope(position_ids.device):
        angles = position_ids.float()[..., None] * inverse_frequencies[None, None]
        angles = torch.cat((angles, angles), -1)
        return angles.cos(), angles.sin()


def rope(inputs, cosine, sine):
    _bf16(inputs, 'RoPE input')
    _require(inputs.ndim == 4 and cosine.shape == sine.shape and cosine.ndim == 3
             and cosine.shape[0] in (1, inputs.shape[0]) and cosine.shape[1:] == (inputs.shape[2], inputs.shape[3])
             and inputs.shape[-1] % 2 == 0, 'RoPE expects BHLD and BLD coefficients')
    _require(cosine.dtype == sine.dtype == torch.float32 and cosine.device == sine.device == inputs.device
             and not cosine.requires_grad and not sine.requires_grad, 'RoPE coefficients must be constant FP32')
    with _scope(inputs.device):
        fp32 = inputs.float()
        first, second = fp32.chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
        return (fp32 * cosine[:, None] + rotated * sine[:, None]).bfloat16()


def swiglu(gate, up):
    _bf16(gate, 'SwiGLU gate')
    _bf16(up, 'SwiGLU up')
    _require(gate.shape == up.shape and gate.device == up.device, 'SwiGLU operands differ')
    with _scope(gate.device):
        fp32 = gate.float()
        return (fp32 * torch.sigmoid(fp32) * up.float()).bfloat16()


def attention_fp32(query, key, value, *, causal=True, scale=None):
    """FP32 mathematical attention diagnostic; explicitly not FA emulation."""
    for name, tensor in (('query', query), ('key', key), ('value', value)):
        _bf16(tensor, name)
        _require(tensor.ndim == 4, 'attention expects BHLD tensors')
    _require(query.device == key.device == value.device and key.shape == value.shape
             and query.shape[0] == key.shape[0] and query.shape[-1] == key.shape[-1]
             and query.shape[1] % key.shape[1] == 0, 'attention Q/K/V dimensions differ')
    _require(type(causal) is bool and (not causal or query.shape[2] == key.shape[2]),
             'causal attention requires equal query/key lengths; no cache supported')
    scale = query.shape[-1] ** -.5 if scale is None else scale
    _require(isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0, 'invalid attention scale')
    _check_fp32_gemm(query.device)
    with _scope(query.device):
        group = query.shape[1] // key.shape[1]
        k = key.float().repeat_interleave(group, dim=1)
        v = value.float().repeat_interleave(group, dim=1)
        scores = query.float() @ k.transpose(-1, -2) * scale
        if causal:
            blocked = torch.ones(query.shape[2], key.shape[2], dtype=torch.bool, device=query.device).triu(1)
            scores = scores.masked_fill(blocked, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        return (probabilities @ v).bfloat16()
