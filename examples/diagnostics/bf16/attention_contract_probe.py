"""Independent one-tile FA2 arithmetic diagnostic, not an acceptance oracle.

This module's formulas import only PyTorch. No Slime/Megatron/TE/FlashAttention
function or native captured output contributes to an expected value. The CLI
imports TE only to obtain the separately measured actual tensors.

Scope: one sequence, THD, equal Q/K lengths 1..128, D=32, no dropout or bias,
causal mask, BF16 Q/K/V/dO; query heads are contiguous groups per KV head.
The confirmed native kernel for the original L=64 control has 128x128 tiles.
Other tile choices, long sequences, CP, and kernel reduction orders are outside
this diagnostic. Explicit casts follow FA2 v2.7.4.post1 source. FP32 matmuls,
exp2/log functions, and reduction trees do not reproduce its CUDA instructions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Boundaries:
    forward_exp_bf16: bool = False
    backward_recompute_and_output_delta: bool = False
    backward_p_bf16: bool = False
    backward_ds_bf16: bool = False
    gqa_per_query_head_bf16: bool = False


VARIANTS = {
    'math_fp32': Boundaries(),
    'forward_exp_bf16': Boundaries(True),
    'recompute_and_output_delta': Boundaries(True, True),
    'backward_p_bf16': Boundaries(True, True, True),
    'backward_p_ds_bf16': Boundaries(True, True, True, True),
    'fa2_contract': Boundaries(True, True, True, True, True),
}


def _f32_fma(a, b, c):
    # Evaluate one multiply-add in FP64 then round once. This isolates the FMA
    # boundary for these bounded FP32 operands; it is not a CUDA instruction
    # emulator or an assertion about exp2f/__logf correctly rounded results.
    return (a.double() * b.double() + c.double()).float()


def _thd(value):
    return value.permute(1, 0, 2).contiguous()


def _validate(query, key, value, upstream):
    if not all(x.dtype == torch.bfloat16 and x.ndim == 3 for x in (query, key, value, upstream)):
        raise ValueError('Expected BF16 THD tensors')
    if not all(x.device == query.device for x in (key, value, upstream)):
        raise ValueError('Devices differ')
    length, heads, width = query.shape
    if not (1 <= length <= 128 and width == 32 and heads > 0 and key.shape[1] > 0):
        raise ValueError('One-tile diagnostic requires 1 <= length <= 128 and D=32')
    if (key.shape != value.shape or key.shape[0] != length or key.shape[2] != width
            or upstream.shape != query.shape or heads % key.shape[1]):
        raise ValueError('Only equal-length causal GQA with divisible heads is supported')
    if query.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Disable TF32 for independent FP32 matmul diagnostics')
    if not all(bool(torch.isfinite(x).all()) for x in (query, key, value, upstream)):
        raise ValueError('Inputs must be finite')


@torch.no_grad()
def evaluate_attention(query, key, value, upstream, *, variant='fa2_contract', return_intermediates=False):
    """Return explicit forward and first-order VJPs without native output input.

    These backward formulas represent the specified floating-point algorithm,
    not finite differences of the discontinuous BF16-rounded forward map.
    """
    _validate(query, key, value, upstream)
    try:
        flags = VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f'Unknown arithmetic variant: {variant}') from error
    with torch.autocast(device_type=query.device.type, enabled=False):
        q, k, v, do = [x.permute(1, 0, 2).float() for x in (query, key, value, upstream)]
        length, heads, width = query.shape
        kv_heads = key.shape[1]
        groups = heads // kv_heads
        k = k.repeat_interleave(groups, 0)
        v = v.repeat_interleave(groups, 0)
        raw_score = q @ k.transpose(-1, -2)
        mask = torch.arange(length, device=q.device)[None, :] > torch.arange(length, device=q.device)[:, None]
        raw_score = raw_score.masked_fill(mask, -torch.inf)
        # The C++ binding takes softmax_scale as float; the log2 multiplier is
        # assigned to float after multiplying by the double M_LOG2E constant.
        scale = torch.tensor(width ** -.5, device=q.device, dtype=torch.float32)
        scale_log2 = (scale.double() * math.log2(math.e)).float()
        log2e = torch.tensor(math.log2(math.e), device=q.device, dtype=torch.float32)
        maximum = raw_score.max(-1, keepdim=True).values
        exponent = torch.exp2(_f32_fma(raw_score, scale_log2, -(maximum * scale_log2)))
        denominator = exponent.sum(-1, keepdim=True)
        lse = _f32_fma(maximum, scale, denominator.log())
        source_output = ((exponent.bfloat16().float() @ v) * denominator.reciprocal()).bfloat16()

        # A separate ordinary FP32 mathematical path makes ablation zero
        # comparable to the existing independent attention control.
        score = raw_score * scale
        ordinary_exp = (score - score.max(-1, keepdim=True).values).exp()
        ordinary_p = ordinary_exp / ordinary_exp.sum(-1, keepdim=True)
        ordinary_output = (ordinary_p @ v).bfloat16()
        output = source_output if flags.forward_exp_bf16 else ordinary_output
        dp = do @ v.transpose(-1, -2)  # FP32 accumulator, no BF16 dP store.
        if flags.backward_recompute_and_output_delta:
            probability = torch.exp2(_f32_fma(raw_score, scale_log2, -(lse * log2e)))
            delta = (do * output.float()).sum(-1, keepdim=True)
        else:
            probability = ordinary_p
            delta = (dp * probability).sum(-1, keepdim=True)
        ds = probability * (dp - delta)  # No scale before the BF16 dS store.
        p_gemm = probability.bfloat16().float() if flags.backward_p_bf16 else probability
        ds_gemm = ds.bfloat16().float() if flags.backward_ds_bf16 else ds
        dq = ((ds_gemm @ k) * scale).bfloat16()
        dk_per_head = (ds_gemm.transpose(-1, -2) @ q) * scale
        dv_per_head = p_gemm.transpose(-1, -2) @ do
        if flags.gqa_per_query_head_bf16:
            dk_per_head = dk_per_head.bfloat16().float()
            dv_per_head = dv_per_head.bfloat16().float()
        dk = dk_per_head.reshape(kv_heads, groups, length, width).sum(1).bfloat16()
        dv = dv_per_head.reshape(kv_heads, groups, length, width).sum(1).bfloat16()
        result = {'forward': _thd(output), 'q_gradient': _thd(dq),
                  'k_gradient': _thd(dk), 'v_gradient': _thd(dv)}
        if return_intermediates:
            return result, {'raw_score': raw_score, 'exponent': exponent, 'denominator': denominator,
                            'lse': lse, 'probability': probability, 'dp': dp, 'delta': delta, 'ds': ds,
                            'dk_per_head': dk_per_head, 'dv_per_head': dv_per_head}
        return result


@torch.no_grad()
def fp64_mathematical_reference(query, key, value, upstream):
    """Independent high-precision analytic softmax VJP; no FA boundary logic."""
    _validate(query, key, value, upstream)
    q, k, v, do = [x.permute(1, 0, 2).double() for x in (query, key, value, upstream)]
    length, heads, width = query.shape
    groups, kv_heads = heads // key.shape[1], key.shape[1]
    k, v = k.repeat_interleave(groups, 0), v.repeat_interleave(groups, 0)
    score = (q @ k.transpose(-1, -2)) / math.sqrt(width)
    mask = torch.ones(length, length, device=q.device, dtype=torch.bool).triu(1)
    score.masked_fill_(mask, -torch.inf)
    e = (score - score.max(-1, keepdim=True).values).exp()
    p = e / e.sum(-1, keepdim=True)
    dp = do @ v.transpose(-1, -2)
    ds = p * (dp - (p * dp).sum(-1, keepdim=True)) / math.sqrt(width)
    result = {'forward': p @ v, 'q_gradient': ds @ k,
              'k_gradient': (ds.transpose(-1, -2) @ q).reshape(kv_heads, groups, length, width).sum(1),
              'v_gradient': (p.transpose(-1, -2) @ do).reshape(kv_heads, groups, length, width).sum(1)}
    return {name: _thd(tensor) for name, tensor in result.items()}


def run_probe(directory):
    # Only this runner imports a native implementation, for actual measurements.
    # All six independent predictions are computed before invoking it.
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import transformer_engine.pytorch as te

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'result.json').exists() or (destination / 'values.npz').exists():
        raise FileExistsError('Refusing to overwrite an existing probe result')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows, arrays, kernels = [], {}, set()

    def metric(actual, expected):
        a, e = actual.double(), expected.double()
        delta = a - e
        return {'elements': a.numel(), 'equal_elements': int((a == e).sum()),
                'max_abs': delta.abs().max().item(), 'absolute_l2': delta.norm().item(),
                'reference_l2': e.norm().item(), 'relative_l2': (delta.norm() / e.norm().clamp_min(1e-300)).item(),
                'finite': bool(torch.isfinite(a).all() and torch.isfinite(e).all())}

    for seed in (101, 173, 271):
        torch.manual_seed(seed)
        q, k, v, do = [torch.randn(shape, device='cuda').mul(scale).bfloat16() for shape, scale in
                        (((64, 4, 32), 1.), ((64, 2, 32), 1.), ((64, 2, 32), 1.), ((64, 4, 32), .1))]
        expected = {name: evaluate_attention(q, k, v, do, variant=name) for name in VARIANTS}
        fp64 = fp64_mathematical_reference(q, k, v, do)
        _, intermediate = evaluate_attention(q, k, v, do, return_intermediates=True)
        for name, tensor in zip(('q', 'k', 'v', 'do'), (q, k, v, do)):
            arrays[f'{seed}_input_{name}'] = tensor.float().cpu().numpy()
        for name, tensor in intermediate.items():
            arrays[f'{seed}_independent_{name}'] = tensor.cpu().numpy()
        qa, ka, va = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        attention = te.DotProductAttention(num_attention_heads=4, kv_channels=32, num_gqa_groups=2,
            attention_dropout=0., qkv_format='thd', attn_mask_type='padding_causal', tp_size=1,
            tp_group=None, cp_group=None, sequence_parallel=False, layer_number=1,
            softmax_scale=32**-.5).cuda().train()
        cu = torch.tensor([0, 64], device='cuda', dtype=torch.int32)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            out = attention(qa, ka, va, cu_seqlens_q=cu, cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
                max_seqlen_q=64, max_seqlen_kv=64, attn_mask_type='padding_causal',
                pad_between_seqs=False).reshape(64, 4, 32)
            out.backward(do)
        kernels.update(event.name for event in profiler.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA)
        actual = dict(forward=out.detach(), q_gradient=qa.grad, k_gradient=ka.grad, v_gradient=va.grad)
        for role in actual:
            arrays[f'{seed}_native_{role}'] = actual[role].float().cpu().numpy()
            arrays[f'{seed}_fp64_{role}'] = fp64[role].cpu().numpy()
            for variant in VARIANTS:
                prediction = expected[variant][role]
                arrays[f'{seed}_{variant}_{role}'] = prediction.float().cpu().numpy()
                rows.append({'seed': seed, 'variant': variant, 'role': role,
                             'native_vs_independent': metric(actual[role], prediction),
                             'independent_vs_fp64': metric(prediction, fp64[role]),
                             'native_vs_fp64': metric(actual[role], fp64[role])})
        print(json.dumps({'seed': seed, 'comparisons': len(rows)}), flush=True)
    raw = destination / 'values.npz'
    np.savez(raw, **arrays)
    record = {'scope': 'One-tile FA2 source-boundary diagnostic, no acceptance policy',
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'variants': {name: asdict(flags) for name, flags in VARIANTS.items()},
              'torch': torch.__version__, 'cuda_capability': list(torch.cuda.get_device_capability()),
              'precision': {'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'matmul_precision': torch.get_float32_matmul_precision()},
              'rows': rows, 'cuda_kernel_names': sorted(kernels),
              'raw': {'file': raw.name, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                      'bytes': raw.stat().st_size}}
    (destination / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    run_probe(arguments.output)
