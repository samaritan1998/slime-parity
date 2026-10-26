"""Private, independent BF16 Qwen3-MoE arithmetic diagnostics.

No native modules, captured routing choices or native outputs are used. The
frozen public HF runner still owns parameter mapping, FP32 optimizer masters,
tokens, masks, global normalization and optimizer updates. Profiles change
arithmetic explicitly; none is a qualified replacement acceptance oracle.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import torch
import torch.nn.functional as F

from training_parity.common import digest, require
import bf16_ops as ops
import contract_reference as dense


DENSE_SHA256 = 'f86e608f2d5dbffb354865be72ece1e28ba480e4f0060303f1071cd1c2a7bb83'
PROFILES = ('bf16_legacy', 'router_fp32', 'weighted_experts', 'packed_projections',
            'fused_norm_rope_swiglu', 'math_attention', 'all_ops_math')


def router_linear(inputs, weight):
    """BF16 operands, strict FP32 projection, BF16 dx/dw writeback.

    The inspected native runtime has no TE wrapper and takes torch.mm's FP32
    fallback. If that wrapper becomes available, TE can request FAST_TF32
    independently of torch's switch; this reference does not claim that kernel.
    """
    require(inputs.dtype == torch.bfloat16 and weight.dtype == torch.float32
            and inputs.ndim == 2 and weight.ndim == 2 and inputs.shape[-1] == weight.shape[-1]
            and inputs.device == weight.device, 'Router requires BF16 inputs and FP32 master weights')
    require(inputs.device.type != 'cuda' or not torch.backends.cuda.matmul.allow_tf32,
            'Disable TF32 for the independent FP32 router diagnostic')
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return F.linear(inputs.float(), weight.bfloat16().float())


def weighted_swiglu(gate, up, probabilities):
    """Explicit FP32 SiLU/product/probability, then one BF16 output boundary.

    This is an arithmetic hypothesis for the compiled weighted activation.
    Native generated-code fusion and reduction/FMA order need GPU evidence.
    """
    require(gate.dtype == up.dtype == torch.bfloat16 and probabilities.dtype == torch.float32
            and gate.ndim == up.ndim == 2 and gate.shape == up.shape
            and probabilities.shape == (gate.shape[0], 1)
            and gate.device == up.device == probabilities.device, 'Weighted SwiGLU dimensions/dtypes differ')
    with torch.autocast(device_type=gate.device.type, enabled=False):
        x = gate.float()
        return (x * torch.sigmoid(x) * up.float() * probabilities).bfloat16()


def packed_expert(module, inputs, probabilities):
    """One gate/up GEMM and one BF16 dinput, retaining the HF parameter map."""
    combined = torch.cat((module.gate_proj.weight, module.up_proj.weight), dim=0)
    gate, up = ops.linear(inputs, combined).chunk(2, dim=-1)
    return module.down_proj(weighted_swiglu(gate, up, probabilities))


def packed_qkv(module, inputs):
    """Group-major [Q heads in group, K, V] projection without native code."""
    dim = module.head_dim
    groups = module.k_proj.weight.shape[0] // dim
    per_group = module.q_proj.weight.shape[0] // (groups * dim)
    width = inputs.shape[-1]
    require(groups > 0 and per_group > 0 and module.v_proj.weight.shape == module.k_proj.weight.shape,
            'Packed QKV dimensions differ')
    weights = torch.cat((module.q_proj.weight.reshape(groups, per_group, dim, width),
                         module.k_proj.weight.reshape(groups, 1, dim, width),
                         module.v_proj.weight.reshape(groups, 1, dim, width)), dim=1)
    projected = ops.linear(inputs, weights.reshape(-1, width))
    projected = projected.reshape(*inputs.shape[:-1], groups, (per_group + 2) * dim)
    query, key, value = projected.split((per_group * dim, dim, dim), dim=-1)
    return (query.reshape(*inputs.shape[:-1], groups * per_group, dim), key, value)


def _tensor_hash(value):
    raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256(raw).hexdigest()


class _State(dense._State):
    def __init__(self, ctx, profile):
        super().__init__(ctx, profile)
        self.dynamic_expected, self.expert_modules = Counter(), {}
        self.pending_routes, self.route_layers = {}, []
        self.metadata.update({
            'model_family': 'qwen3_moe', 'dense_control_sha256': DENSE_SHA256,
            'routing_steps': [], 'routing_source': 'independent current HF hidden states and effective weights',
            'limits': [
                'Original policy and numerical failures are unchanged; this is not an acceptance oracle.',
                'The inspected native runtime takes the torch.mm router fallback because its TE wrapper is unavailable. A changed runtime must recheck this path; an available TE wrapper can request FAST_TF32 independently of torch.allow_tf32.',
                'Weighted SwiGLU uses an explicit FP32 local formula; compiled native fusion/FMA order is not reproduced.',
                'Independent GEMMs do not reproduce TE grouped-GEMM tiling/reduction order.',
                'Math attention is not FA2/FA3; other profiles retain HF eager attention.',
                'Expert counts include every forward token, including unsupervised positions; they do not prove nonzero expert gradients.',
                'Unused experts in one microbatch remain unused. An expert unused for the whole step is rejected by the public runner; no zero gradients are manufactured.',
                'Single-device Qwen3-MoE reference only; EP/ETP collective rounding remains a candidate-side comparison.',
            ],
        })

    def install_forward(self, module, name, strategy, function):
        super().install_forward(module, name, strategy, function)
        if '.experts.' in name:
            prefix = name.split('.experts.')[0] + '.experts.' + name.split('.experts.')[1].split('.')[0]
            self.expert_modules.setdefault(prefix, []).append(name)
            del self.expected[name]

    def routes(self, name, module, inputs, logits, selection_scores, selected, probabilities, effective):
        require(self.micro > 0 and inputs.ndim == 2, 'MoE routing has no sample context')
        sample = self.ctx.suite['description']['inputs'][self.step]['samples'][self.micro - 1]
        length, experts, topk = len(sample['tokens']), module.num_experts, module.top_k
        require(logits.shape == (length, experts) and selected.shape == probabilities.shape == (length, topk)
                and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probabilities).all()),
                'MoE routing shape/nonfinite evidence differs')
        counts = torch.bincount(selected.detach().reshape(-1), minlength=experts).cpu().tolist()
        for expert, count in enumerate(counts):
            if count:
                self.dynamic_expected.update(self.expert_modules[name + '.experts.' + str(expert)])
        ordered_logits = torch.topk(logits.detach().float(), topk + 1, dim=-1).values
        ordered_scores = torch.topk(selection_scores.detach().float(), topk + 1, dim=-1).values
        margins = ordered_logits[:, topk - 1] - ordered_logits[:, topk]
        record = {
            'sample_id': sample['id'], 'token_positions': list(range(length)),
            'next_token_loss_mask': sample['loss_mask'][1:] + [0],
            'input_dtype': str(inputs.dtype), 'input_shape': list(inputs.shape), 'input_sha256': _tensor_hash(inputs),
            'logit_dtype': str(logits.dtype), 'logit_sha256': _tensor_hash(logits),
            'logits': logits.detach().float().cpu().tolist(),
            'selected_experts': selected.detach().cpu().tolist(),
            'probabilities_fp32': probabilities.detach().float().cpu().tolist(),
            'effective_probabilities_dtype': str(effective.dtype),
            'effective_probabilities': effective.detach().float().cpu().tolist(),
            'logit_cut_margins': margins.cpu().tolist(),
            'selection_score_cut_margins': (ordered_scores[:, topk - 1] - ordered_scores[:, topk]).cpu().tolist(),
            'expert_token_counts': counts,
        }
        self.pending_routes.setdefault(name, []).append(record)

    def finish_step(self, step):
        samples = self.ctx.suite['description']['inputs'][step]['samples']
        expected = Counter({key: value * len(samples) for key, value in self.expected.items()})
        expected.update(self.dynamic_expected)
        require(step == self.step and self.micro == len(samples) and self.counts == expected,
                f'MoE arithmetic operator coverage differs at step {step}')
        require(set(self.pending_routes) == set(self.route_layers), 'MoE routing layer coverage differs')
        summary = {}
        for name in self.route_layers:
            records = self.pending_routes[name]
            require([record['sample_id'] for record in records] == [sample['id'] for sample in samples]
                    and all(record['token_positions'] == list(range(len(sample['tokens'])))
                            for record, sample in zip(records, samples)), 'MoE routing sample/token coverage differs')
            counts = [sum(row['expert_token_counts'][index] for row in records)
                      for index in range(len(records[0]['expert_token_counts']))]
            margins = [margin for record in records for margin in record['logit_cut_margins']]
            summary[name] = {'tokens': len(margins), 'expert_token_counts': counts,
                             'all_experts_selected': all(count > 0 for count in counts),
                             'minimum_logit_cut_margin': min(margins), 'zero_logit_cut_margins': margins.count(0.0)}
        record = {'profile': self.profile, 'microbatches': len(samples),
                  'sample_ids': [sample['id'] for sample in samples],
                  'operator_calls': dict(sorted(self.counts.items())), 'routing': summary}
        self.metadata['steps'].append(deepcopy(record))
        self.metadata['routing_steps'].append({'step': step, 'layers': self.pending_routes})
        self.step += 1
        self.micro = 0
        self.counts.clear()
        self.dynamic_expected.clear()
        self.pending_routes = {}
        return record

    def close(self):
        if self.pending_routes:
            self.metadata['incomplete_routing_step'] = {
                'step': self.step, 'microbatches_seen': self.micro, 'layers': self.pending_routes,
                'completed_optimizer_step': False,
            }
        super().close()


def _install(model, state, qwen):
    require(state.loaded_model is None and type(model) is qwen.Qwen3MoeForCausalLM,
            'MoE arithmetic control requires exactly one Qwen3MoeForCausalLM')
    state.loaded_model = model
    config = model.config
    require(config.model_type == 'qwen3_moe' and not config.tie_word_embeddings and not config.attention_bias
            and config.hidden_act == 'silu' and config.rope_scaling is None
            and not config.use_sliding_window and config.sliding_window is None
            and config.decoder_sparse_step == 1 and not config.mlp_only_layers
            and config.num_experts_per_tok == 2 and config.num_experts > 2 and config.norm_topk_prob is True
            and config.router_aux_loss_coef == 0 and config.output_router_logits is False
            and config._attn_implementation == 'eager' and not model.model.gradient_checkpointing,
            'Unsupported Qwen3-MoE arithmetic configuration')
    named = list(model.named_parameters(remove_duplicate=False))
    require(len({id(parameter) for _, parameter in named}) == len(named), 'Shared MoE parameters unsupported')
    require(all(p.requires_grad and p.dtype == torch.float32 for _, p in named), 'MoE requires trainable FP32 Parameter storage')
    level = PROFILES.index(state.profile)
    native_route, weighted, packed, advanced, mathematical = (level >= n for n in (1, 2, 3, 4, 5))
    state.metadata['routing_contract'] = {
        'projection': 'BF16 operands to strict FP32 logits, BF16 dx/dw' if native_route else 'HF BF16 projection and gradients',
        'selection': 'topk logits then FP32 selected softmax' if native_route else 'FP32 full softmax, topk probabilities, selected renormalization',
        'weighting': 'FP32 probability inside SwiGLU before fc2, BF16 activation' if weighted else 'BF16 probability times BF16 expert output after fc2',
        'dispatch_order': 'expert-major, ascending token index' if weighted else 'expert-major, topk-slot-major then token index',
        'expert_fc1': 'one packed gate/up GEMM' if weighted else 'separate gate/up projections',
        'attention_qkv': 'one group-major Q/K/V GEMM' if packed else 'separate Q/K/V projections',
        'forced_routes': False, 'topk': 2, 'num_experts': config.num_experts,
    }
    modeled_parameters, attention_names = set(), {}

    def sparse_forward(name):
        def forward(module, hidden):
            batch, length, width = hidden.shape
            require(batch == 1 and hidden.dtype == torch.bfloat16, 'MoE diagnostic expects one BF16 sample per microbatch')
            inputs = hidden.view(-1, width)
            logits = module.gate(inputs)
            if native_route:
                selected_logits, selected = torch.topk(logits, module.top_k, dim=-1)
                probabilities = F.softmax(selected_logits, dim=-1, dtype=torch.float32)
                selection_scores = logits
            else:
                selection_scores = F.softmax(logits, dim=1, dtype=torch.float32)
                probabilities, selected = torch.topk(selection_scores, module.top_k, dim=-1)
                probabilities /= probabilities.sum(dim=-1, keepdim=True)
            effective = probabilities if weighted else probabilities.to(inputs.dtype)
            state.routes(name, module, inputs, logits, selection_scores, selected, probabilities, effective)
            output = torch.zeros_like(inputs)
            if weighted:
                for index, expert in enumerate(module.experts):
                    rows, slots = torch.where(selected == index)
                    if rows.numel():
                        current = inputs[rows]
                        values = expert(current, effective[rows, slots, None])
                        output.index_add_(0, rows, values)
            else:
                # Preserve HF's slot-major token order and aggregation for the
                # legacy identity control, including only invoking hit experts.
                expert_mask = F.one_hot(selected, num_classes=module.num_experts).permute(2, 1, 0)
                for index in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
                    slots, rows = torch.where(expert_mask[index].squeeze(0))
                    current = inputs[None, rows].reshape(-1, width)
                    values = module.experts[index](current) * effective[rows, slots, None]
                    output.index_add_(0, rows, values.to(inputs.dtype))
            return output.reshape(batch, length, width), logits
        return forward

    def attention_forward(module, hidden_states, position_embeddings, attention_mask=None,
                          past_key_values=None, cache_position=None, **kwargs):
        require(past_key_values is None, 'MoE arithmetic control does not support a KV cache')
        query, key, value = packed_qkv(module, hidden_states)
        query = module.q_norm(query).transpose(1, 2)
        key = module.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = qwen.apply_rotary_pos_emb(query, key, *position_embeddings)
        output, weights = qwen.eager_attention_forward(module, query, key, value, attention_mask,
                                                        scaling=module.scaling, dropout=0.0)
        return module.o_proj(output.reshape(*hidden_states.shape[:-1], -1).contiguous()), weights

    for name, module in model.named_modules():
        if type(module) is torch.nn.Embedding:
            require(module.max_norm is None and not module.scale_grad_by_freq and not module.sparse,
                    'MoE embedding renorm/frequency scaling/sparse gradients unsupported')
            function = (lambda m, x: ops.embedding(x, m.weight, m.padding_idx)) if mathematical else dense._legacy_embedding
            state.install_forward(module, name, 'chunk32_bf16_gradient' if mathematical else 'bf16_embedding_autograd', function)
            modeled_parameters.add(name + '.weight')
        elif type(module) is torch.nn.Linear:
            require(module.bias is None, 'MoE arithmetic control requires bias-free linears')
            modeled_parameters.add(name + '.weight')
            if weighted and '.experts.' in name and name.endswith(('.gate_proj', '.up_proj')):
                state.strategies[name] = 'weight_in_joint_expert_fc1'
            elif packed and '.self_attn.' in name and name.endswith(('.q_proj', '.k_proj', '.v_proj')):
                state.strategies[name] = 'weight_in_group_major_joint_qkv'
            elif name.endswith('.mlp.gate'):
                function = (lambda m, x: router_linear(x, m.weight)) if native_route else dense._legacy_linear
                state.install_forward(module, name, 'strict_fp32_router_bf16_dx_dw' if native_route else 'bf16_router', function)
            else:
                function = (lambda m, x: ops.linear(x, m.weight)) if weighted else dense._legacy_linear
                state.install_forward(module, name, 'fp32_wgrad' if weighted else 'bf16_wgrad', function)
        elif type(module) is qwen.Qwen3MoeSparseMoeBlock:
            state.route_layers.append(name)
            state.install_forward(module, name, 'independent_topk_expert_dispatch', sparse_forward(name))
        elif type(module) is qwen.Qwen3MoeMLP:
            require('.experts.' in name, 'Non-expert MLP is outside this MoE diagnostic')
            function = packed_expert if weighted else dense._legacy_swiglu
            state.install_forward(module, name, 'joint_fc1_fp32_weighted_swiglu_before_fc2' if weighted else 'hf_expert', function)
        elif type(module) is qwen.Qwen3MoeRMSNorm:
            function = (lambda m, x: ops.rmsnorm(x, m.weight, m.variance_epsilon)) if advanced else dense._legacy_norm
            state.install_forward(module, name, 'fp32_rms_product_bf16_gamma_grad' if advanced else 'hf_bf16_norm', function)
            modeled_parameters.add(name + '.weight')
        elif type(module) is qwen.Qwen3MoeRotaryEmbedding:
            state.install_forward(module, name, 'fp32_rope_coefficients' if advanced else 'bf16_rope_coefficients',
                                  lambda m, x, position_ids: dense._coefficients(m, x, position_ids, advanced))
        elif type(module) is qwen.Qwen3MoeAttention:
            require(module.attention_dropout == 0 and module.sliding_window is None, 'MoE attention dropout/window unsupported')
            attention_names[id(module)] = name
            if packed:
                state.install_forward(module, name, 'joint_group_major_qkv_fp32_wgrad', attention_forward)
    require(modeled_parameters == {name for name, _ in named}, 'Unmodeled trainable MoE parameter')
    require(len(attention_names) == len(state.route_layers) == config.num_hidden_layers, 'MoE layer coverage differs')
    state.expected['model_forward'] = 1
    state.expected['rope_apply'] = config.num_hidden_layers
    state.strategies['rope_apply'] = 'fp32_rotation_then_bf16' if advanced else 'bf16_rotation'

    def before_forward(module, args, kwargs):
        tokens = kwargs.get('input_ids', args[0] if args else None)
        require(tokens is not None and kwargs.get('use_cache') is False, 'MoE diagnostic requires tokens and no cache')
        state.set_sample_context(tokens)

    state.handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True, prepend=True))

    def apply_rope(query, key, cosine, sine, position_ids=None, unsqueeze_dim=1):
        require(unsqueeze_dim == 1, 'MoE arithmetic control expects BHLD RoPE')
        state.counts['rope_apply'] += 1
        if advanced:
            return ops.rope(query, cosine, sine), ops.rope(key, cosine, sine)
        cosine, sine = cosine[:, None], sine[:, None]

        def rotate(value):
            first, second = value.chunk(2, -1)
            return value * cosine + torch.cat((-second, first), -1) * sine

        return rotate(query), rotate(key)

    state.replace(qwen, 'apply_rotary_pos_emb', apply_rope)
    if mathematical:
        for name in attention_names.values():
            state.expected[name + '.math_attention'] = 1
            state.strategies[name + '.math_attention'] = 'fp32_scores_softmax_pv_then_bf16_not_flashattention'

        def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            require(id(module) in attention_names and dropout == 0, 'Unexpected MoE attention/dropout')
            state.counts[attention_names[id(module)] + '.math_attention'] += 1
            length = query.shape[2]
            require(key.shape[2] == length, 'MoE unequal-length attention unsupported')
            if attention_mask is not None:
                require(attention_mask.shape[-2:] == (length, length) and attention_mask.shape[-3] == 1,
                        'MoE attention mask dimensions differ')
                blocked = torch.ones(length, length, device=query.device, dtype=torch.bool).triu(1)
                require(bool((attention_mask[..., ~blocked] == 0).all())
                        and bool((attention_mask[..., blocked] < -1e20).all()), 'MoE attention requires full causal mask')
            return ops.attention_fp32(query, key, value, causal=True, scale=scaling).transpose(1, 2).contiguous(), None

        state.replace(qwen, 'eager_attention_forward', eager_attention)
        state.metadata['limits'].append('Math profile also replaces embedding with the explicit chunk32 backward formula.')


def run_reference(ctx):
    import transformers
    from transformers import AutoModelForCausalLM
    import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen
    from slime_parity.adapters import hf

    options = ctx.stage['configuration']['options']
    require(isinstance(options, dict) and set(options) <= {'profile', *dense.PUBLIC_OPTIONS}, 'Unknown MoE arithmetic option')
    requested = options.get('profile', 'bf16_legacy')
    require(requested in PROFILES, 'Unknown MoE arithmetic profile')
    require(all(type(options.get(key, value)) is type(value) and options.get(key, value) == value
                for key, value in dense.PUBLIC_OPTIONS.items()), 'MoE requires FP32 storage/outer compute, microbatch 1 and late normalization')
    require(ctx.suite['training']['dtype'] == 'bfloat16' and transformers.__version__ == '4.57.6',
            'MoE arithmetic control requires BF16 training and Transformers 4.57.6')
    require(digest(Path(hf.__file__)) == dense.HF_SHA256 and digest(Path(ops.__file__)) == dense.OPS_SHA256
            and digest(Path(dense.__file__)) == DENSE_SHA256, 'Frozen MoE dependency hash changed')
    state = _State(ctx, 'math_attention' if requested == 'all_ops_math' else requested)
    state.metadata['requested_profile'] = requested
    state.metadata['qwen_source_sha256'] = digest(Path(qwen.__file__))
    proxy = dense._Context(ctx, state)
    original_load = AutoModelForCausalLM.from_pretrained
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
                'MoE arithmetic control did not complete every optimizer step')
    finally:
        state.close()
