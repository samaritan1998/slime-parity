"""Private dense Qwen3 BF16 arithmetic controls over the frozen public runner.

The public runner owns weights, mapping, tokens, objective, FP32 masters and
optimizer updates. These controls change only selected forward/backward
arithmetic. They remain hypotheses under the original acceptance policy.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from training_parity.common import digest, require
import bf16_ops as ops


HF_SHA256 = '48b96a757a280a96e256c19485ecda85a650dc6afaf5fe8928b37708df5c8cf8'
OPS_SHA256 = 'd4e0eadb72cdd35ea53090e10b684784b375bf0d0882d0f4200d0676ea18b42f'
PUBLIC_OPTIONS = {'compute_dtype': 'float32', 'parameter_dtype': 'float32',
                  'micro_batch_size': 1, 'normalization_timing': 'after_backward'}
PROFILES = ('bf16_legacy', 'linear_fp32wgrad', 'fused_norm_rope_swiglu', 'math_attention', 'all_ops_math')


class _Context:
    """Preserve the frozen caller configuration while exposing public controls."""
    def __init__(self, ctx, state):
        self.original, self.state = ctx, state
        self.stage = {**ctx.stage, 'configuration': {**ctx.stage['configuration'], 'options': dict(PUBLIC_OPTIONS)}}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def record_step(self, step, **kwargs):
        record = self.state.finish_step(step)
        details = dict(kwargs.get('details') or {})
        details['arithmetic_control'] = record
        self.original.record_step(step, **{**kwargs, 'details': details})


class _State:
    def __init__(self, ctx, profile):
        self.ctx, self.profile = ctx, profile
        self.step, self.micro = 0, 0
        self.counts, self.expected = Counter(), Counter()
        self.strategies, self.restore, self.handles = {}, [], []
        self.loaded_model = None
        self.metadata = {'purpose': 'independent arithmetic diagnostic; not a qualified acceptance oracle',
                         'profile': profile, 'public_runner_options': dict(PUBLIC_OPTIONS),
                         'effective_parameter_dtype': 'bfloat16', 'effective_activation_dtype': 'bfloat16',
                         'hf_source_sha256': HF_SHA256, 'ops_source_sha256': OPS_SHA256,
                         'module_strategies': self.strategies, 'steps': [], 'restored': False,
                         'limits': ['Original policy and failed results are not changed.',
                                    'Math attention is not a FlashAttention implementation or backend contract.',
                                    'FP32 operation order does not reproduce all native fused arithmetic.',
                                    'Dense Qwen3 only; no tied parameters, dropout, cache, sliding windows or RoPE scaling.']}
        ctx.metadata['arithmetic_control'] = self.metadata

    def replace(self, owner, name, value):
        namespace = vars(owner)
        self.restore.append((owner, name, name in namespace, namespace.get(name)))
        setattr(owner, name, value)

    def set_sample_context(self, tokens):
        batches = self.ctx.suite['description']['inputs']
        require(self.step < len(batches), 'Unexpected model call after final step')
        samples = batches[self.step]['samples']
        require(self.micro < len(samples), 'Unexpected extra microbatch')
        sample = samples[self.micro]
        require(tokens.ndim == 2 and tokens.shape[0] == 1 and tokens[0].tolist() == sample['tokens'],
                'Arithmetic diagnostic microbatch differs from frozen tokens/order')
        self.ctx.diagnostic_sample_ids = [sample['id']]
        self.ctx.diagnostic_step = self.step
        self.ctx.diagnostic_microbatch = self.micro
        self.micro += 1
        self.counts['model_forward'] += 1

    def finish_step(self, step):
        require(step == self.step, 'Arithmetic diagnostic optimizer step differs')
        count = len(self.ctx.suite['description']['inputs'][step]['samples'])
        require(self.micro == count and self.counts == Counter({k: v * count for k, v in self.expected.items()}),
                f'Arithmetic diagnostic operator coverage differs at step {step}')
        record = {'profile': self.profile, 'microbatches': count, 'sample_ids': [
            sample['id'] for sample in self.ctx.suite['description']['inputs'][step]['samples']],
                  'operator_calls': dict(sorted(self.counts.items()))}
        self.metadata['steps'].append(deepcopy(record))
        self.step += 1
        self.micro = 0
        self.counts.clear()
        return record

    def install_forward(self, module, name, strategy, function):
        self.strategies[name] = strategy
        self.expected[name] = 1

        def forward(instance, *args, **kwargs):
            self.counts[name] += 1
            device = next(instance.parameters(), None)
            device = device.device if device is not None else instance.inv_freq.device
            with torch.autocast(device_type=device.type, enabled=False):
                return function(instance, *args, **kwargs)

        self.replace(module, 'forward', MethodType(forward, module))

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        for owner, name, existed, value in reversed(self.restore):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.metadata['restored'] = True


def _legacy_norm(module, inputs):
    fp32 = inputs.float()
    normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + module.variance_epsilon)
    return module.weight.bfloat16() * normalized.to(inputs.dtype)


def _legacy_embedding(module, indices):
    return F.embedding(indices, module.weight.bfloat16(), module.padding_idx, module.max_norm,
                       module.norm_type, module.scale_grad_by_freq, module.sparse)


def _legacy_linear(module, inputs):
    return F.linear(inputs, module.weight.bfloat16())


def _legacy_swiglu(module, inputs):
    return module.down_proj(F.silu(module.gate_proj(inputs)) * module.up_proj(inputs))


def _coefficients(module, inputs, position_ids, fp32_output):
    require(module.rope_type == 'default' and module.inv_freq.dtype == torch.float32,
            'Expected default FP32 RoPE frequencies')
    # Retain HF 4.57.6's frequency construction to isolate output precision.
    inverse = module.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(inputs.device)
    frequencies = (inverse @ position_ids[:, None, :].float()).transpose(1, 2)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine, sine = angles.cos() * module.attention_scaling, angles.sin() * module.attention_scaling
    dtype = torch.float32 if fp32_output else inputs.dtype
    return cosine.to(dtype), sine.to(dtype)


def _install(model, state, qwen):
    require(state.loaded_model is None and type(model) is qwen.Qwen3ForCausalLM,
            'Arithmetic control loads exactly one dense Qwen3 causal model')
    state.loaded_model = model
    config = model.config
    require(config.model_type == 'qwen3' and not config.tie_word_embeddings and not config.attention_bias
            and config.hidden_act == 'silu' and config.rope_scaling is None
            and all(kind == 'full_attention' for kind in config.layer_types)
            and config._attn_implementation == 'eager' and not model.model.gradient_checkpointing,
            'Unsupported dense Qwen3 configuration for arithmetic control')
    all_named = list(model.named_parameters(remove_duplicate=False))
    require(len({id(p) for _, p in all_named}) == len(all_named), 'Shared parameters are not supported by arithmetic control')
    require(all(p.requires_grad and p.dtype == torch.float32 for _, p in all_named),
            'Arithmetic control requires all trainable FP32 Parameter storage')
    advanced = state.profile in ('fused_norm_rope_swiglu', 'math_attention')
    math_attention = state.profile == 'math_attention'
    linear_fp32 = state.profile != 'bf16_legacy'
    modeled_parameters, attention_names = set(), {}
    for name, module in model.named_modules():
        if type(module) is torch.nn.Embedding:
            require(module.max_norm is None and not module.scale_grad_by_freq and not module.sparse,
                    'Embedding renorm/frequency scaling/sparse gradients unsupported')
            function = ((lambda m, x: ops.embedding(x, m.weight, m.padding_idx))
                        if math_attention else _legacy_embedding)
            state.install_forward(module, name, 'chunk32_bf16_gradient' if math_attention else 'bf16_embedding_autograd', function)
            modeled_parameters.add(name + '.weight')
        elif type(module) is torch.nn.Linear:
            require(module.bias is None, 'Arithmetic control only supports bias-free linear')
            function = (lambda m, x: ops.linear(x, m.weight)) if linear_fp32 else _legacy_linear
            state.install_forward(module, name, 'fp32_wgrad' if linear_fp32 else 'bf16_wgrad', function)
            modeled_parameters.add(name + '.weight')
        elif type(module) is qwen.Qwen3RMSNorm:
            function = (lambda m, x: ops.rmsnorm(x, m.weight, m.variance_epsilon)) if advanced else _legacy_norm
            state.install_forward(module, name, 'fp32_rms_product_bf16_gamma_grad' if advanced else 'hf_bf16_norm', function)
            modeled_parameters.add(name + '.weight')
        elif type(module) is qwen.Qwen3MLP:
            function = ((lambda m, x: m.down_proj(ops.swiglu(m.gate_proj(x), m.up_proj(x))))
                        if advanced else _legacy_swiglu)
            state.install_forward(module, name, 'fp32_swiglu_then_bf16' if advanced else 'bf16_silu_product', function)
        elif type(module) is qwen.Qwen3RotaryEmbedding:
            state.install_forward(module, name, 'fp32_rope_coefficients' if advanced else 'bf16_rope_coefficients',
                                  lambda m, x, position_ids: _coefficients(m, x, position_ids, advanced))
        elif type(module) is qwen.Qwen3Attention:
            require(module.attention_dropout == 0 and module.sliding_window is None, 'Attention dropout/window unsupported')
            attention_names[id(module)] = name
    require(modeled_parameters == {name for name, _ in all_named}, 'Unmodeled trainable parameter in arithmetic control')
    require(len(attention_names) == config.num_hidden_layers, 'Attention layer coverage differs')
    state.expected['model_forward'] = 1
    state.expected['rope_apply'] = config.num_hidden_layers
    state.strategies['rope_apply'] = 'fp32_rotation_then_bf16' if advanced else 'bf16_rotation'

    def before_forward(module, args, kwargs):
        tokens = kwargs.get('input_ids', args[0] if args else None)
        require(tokens is not None and kwargs.get('use_cache') is False, 'Diagnostic requires token input and no cache')
        state.set_sample_context(tokens)

    state.handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True, prepend=True))

    def apply_rope(query, key, cosine, sine, position_ids=None, unsqueeze_dim=1):
        require(unsqueeze_dim == 1, 'Qwen3 arithmetic control expects BHLD RoPE')
        state.counts['rope_apply'] += 1
        with torch.autocast(device_type=query.device.type, enabled=False):
            if advanced:
                return ops.rope(query, cosine, sine), ops.rope(key, cosine, sine)
            cosine, sine = cosine[:, None], sine[:, None]

            def rotate(value):
                first, second = value.chunk(2, -1)
                return value * cosine + torch.cat((-second, first), -1) * sine

            return rotate(query), rotate(key)

    state.replace(qwen, 'apply_rotary_pos_emb', apply_rope)
    if math_attention:
        for name in attention_names.values():
            state.expected[name + '.math_attention'] = 1
            state.strategies[name + '.math_attention'] = 'fp32_scores_softmax_pv_then_bf16_not_flashattention'

        def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            require(id(module) in attention_names and dropout == 0, 'Unexpected attention module/dropout')
            state.counts[attention_names[id(module)] + '.math_attention'] += 1
            length = query.shape[2]
            require(key.shape[2] == length, 'Cache/unequal-length attention unsupported')
            if attention_mask is not None:
                require(attention_mask.shape[-2:] == (length, length) and attention_mask.shape[-3] == 1,
                        'Unexpected attention-mask dimensions')
                blocked = torch.ones(length, length, device=query.device, dtype=torch.bool).triu(1)
                require(bool((attention_mask[..., ~blocked] == 0).all())
                        and bool((attention_mask[..., blocked] < -1e20).all()),
                        'Arithmetic math attention only accepts the full causal mask')
            with torch.autocast(device_type=query.device.type, enabled=False):
                output = ops.attention_fp32(query, key, value, causal=True, scale=scaling)
            return output.transpose(1, 2).contiguous(), None

        state.replace(qwen, 'eager_attention_forward', eager_attention)


def run_reference(ctx):
    import transformers
    from transformers import AutoModelForCausalLM
    import transformers.models.qwen3.modeling_qwen3 as qwen
    from slime_parity.adapters import hf

    options = ctx.stage['configuration']['options']
    require(isinstance(options, dict) and set(options) <= {'profile', *PUBLIC_OPTIONS}, 'Unknown arithmetic control option')
    requested = options.get('profile', 'bf16_legacy')
    require(requested in PROFILES, 'Unknown arithmetic profile')
    require(all(type(options.get(key, value)) is type(value) and options.get(key, value) == value
                for key, value in PUBLIC_OPTIONS.items()),
            'Arithmetic control requires FP32 storage/outer compute, microbatch 1 and late normalization')
    require(ctx.suite['training']['dtype'] == 'bfloat16', 'Arithmetic control requires BF16 training initialization')
    require(transformers.__version__ == '4.57.6', 'Arithmetic control was reviewed for Transformers 4.57.6')
    require(digest(Path(hf.__file__)) == HF_SHA256 and digest(Path(ops.__file__)) == OPS_SHA256,
            'Arithmetic control implementation hash changed')
    state = _State(ctx, 'math_attention' if requested == 'all_ops_math' else requested)
    state.metadata['requested_profile'] = requested
    state.metadata['qwen_source_sha256'] = digest(Path(qwen.__file__))
    if state.profile == 'math_attention':
        state.metadata['limits'].append('Math profile also changes embedding to explicit verified chunk32 arithmetic; earlier profiles retain torch embedding autograd.')
    proxy = _Context(ctx, state)
    original_load = AutoModelForCausalLM.from_pretrained
    # Preserve absence of instance attributes so a temporary wrapper cannot
    # shadow inherited descriptors after success or a failed model load.
    for name in ('diagnostic_sample_ids', 'diagnostic_step', 'diagnostic_microbatch'):
        state.replace(ctx, name, None)

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        _install(model, state, qwen)
        return model

    state.replace(AutoModelForCausalLM, 'from_pretrained', staticmethod(load))
    try:
        hf.run_reference(proxy)
        require(state.step == len(ctx.suite['description']['inputs']) and state.micro == 0,
                'Arithmetic control did not complete every optimizer step')
    finally:
        state.close()
