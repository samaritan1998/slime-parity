"""Qwen3 RoPE arithmetic diagnostic over the pinned public HF reference.

This changes the reference arithmetic and is not a new acceptance oracle. Only
RoPE coefficient output precision and rotary multiply/add precision change.
No native, Megatron, or Transformer Engine implementation is imported.
"""
from contextlib import contextmanager
from pathlib import Path

from training_parity.common import digest, read, require


PUBLIC_HF_SHA256 = '48b96a757a280a96e256c19485ecda85a650dc6afaf5fe8928b37708df5c8cf8'
EXPECTED_OPTIONS = {'compute_dtype': 'bfloat16', 'parameter_dtype': 'bfloat16',
                    'micro_batch_size': 1, 'normalization_timing': 'after_backward'}


def _apply_rotary_fp32(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Explicit split-half rotation, with one final cast per Q/K output."""
    import torch

    require(unsqueeze_dim in (1, 2) and q.ndim == k.ndim == 4,
            'RoPE diagnostic expects BHLD or BLHD Q/K')
    token_dim = 2 if unsqueeze_dim == 1 else 1
    require(cos.dtype == sin.dtype == torch.float32, 'RoPE coefficients must arrive as FP32')
    require(cos.ndim == 3 and cos.shape == sin.shape and cos.shape[0] in (1, q.shape[0])
            and q.shape[0] == k.shape[0] and cos.shape[1] == q.shape[token_dim] == k.shape[token_dim]
            and cos.shape[2] == q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0,
            'RoPE diagnostic coefficient/token axes differ')
    require(q.device == k.device == cos.device == sin.device, 'RoPE devices differ')
    require(q.dtype in (torch.float32, torch.bfloat16) and k.dtype in (torch.float32, torch.bfloat16),
            'Unexpected RoPE input precision')
    # position_ids was already applied when HF constructed the coefficients;
    # HF's apply_rotary_pos_emb likewise treats that deprecated argument as unused.
    with torch.autocast(device_type=q.device.type, enabled=False):
        cosine, sine = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)

        def rotate(value):
            fp32 = value.float()
            half = fp32.shape[-1] // 2
            rotated = torch.cat((-fp32[..., half:], fp32[..., :half]), dim=-1)
            return (fp32 * cosine + rotated * sine).to(value.dtype)

        return rotate(q), rotate(k)


@contextmanager
def _fp32_rope(qwen, counts):
    """Restore exact original class/module attributes, including on failure."""
    original_forward = qwen.Qwen3RotaryEmbedding.__dict__['forward']
    original_apply = qwen.apply_rotary_pos_emb

    def forward(module, x, position_ids):
        import torch

        require(module.rope_type == 'default' and module.inv_freq.dtype == torch.float32,
                'RoPE diagnostic requires default frequencies preserved in FP32')
        counts['coefficient_calls'] += 1
        # HF uses x only for device and its final coefficient dtype. Its own
        # position IDs, frequency buffer, FP32 trig and scaling remain intact.
        result = original_forward(module, x.float(), position_ids)
        require(all(value.dtype == torch.float32 for value in result), 'HF did not return FP32 trig')
        return result

    def apply(*args, **kwargs):
        counts['rotation_calls'] += 1
        return _apply_rotary_fp32(*args, **kwargs)

    qwen.Qwen3RotaryEmbedding.forward = forward
    qwen.apply_rotary_pos_emb = apply
    try:
        yield
    finally:
        qwen.Qwen3RotaryEmbedding.forward = original_forward
        qwen.apply_rotary_pos_emb = original_apply
        counts['attributes_restored'] = (
            qwen.Qwen3RotaryEmbedding.__dict__['forward'] is original_forward
            and qwen.apply_rotary_pos_emb is original_apply)


def run_reference(ctx):
    import transformers
    import transformers.models.qwen3.modeling_qwen3 as qwen
    from slime_parity.adapters import hf

    require(transformers.__version__ == '4.57.6', 'RoPE diagnostic was reviewed for HF 4.57.6')
    require(digest(Path(hf.__file__)) == PUBLIC_HF_SHA256, 'Expected the pinned public HF reference implementation')
    require(ctx.stage['configuration']['options'] == EXPECTED_OPTIONS,
            'RoPE diagnostic requires the unchanged BF16 microbatch/normalization controls')
    config = read(Path(ctx.plan['artifacts']['hf_checkpoint']['path']) / 'config.json')
    require(config['model_type'] == 'qwen3' and config.get('rope_scaling') is None,
            'RoPE diagnostic is scoped to dense Qwen3 default RoPE')
    counts = {'coefficient_calls': 0, 'rotation_calls': 0, 'attributes_restored': False}
    ctx.metadata['rope_arithmetic_diagnostic'] = {
        'purpose': 'altered reference arithmetic diagnostic; not a replacement acceptance oracle',
        'change': 'HF default RoPE cos/sin remain FP32; split-half Q/K multiply/add in FP32; final output cast to original input dtype',
        'public_hf_sha256': PUBLIC_HF_SHA256,
        'qwen_modeling_sha256': digest(Path(qwen.__file__)),
        'options': dict(EXPECTED_OPTIONS), 'counts': counts,
        'limits': 'Original thresholds and numerical failures remain authoritative. This does not establish TE fused kernel or backward equivalence.',
    }
    with _fp32_rope(qwen, counts):
        hf.run_reference(ctx)
    batches = sum(len(batch['samples']) for batch in ctx.suite['description']['inputs'])
    require(counts['coefficient_calls'] == batches
            and counts['rotation_calls'] == batches * config['num_hidden_layers'],
            'RoPE diagnostic did not observe every expected microbatch/layer')
