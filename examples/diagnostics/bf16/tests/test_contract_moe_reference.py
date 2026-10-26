"""Independent FP64 local checks and CPU three-step MoE integration checks."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen

from slime_parity.adapters import hf
from training_parity.common import EvidenceError, write_new
from contract_moe_reference import packed_qkv, router_linear, run_reference, weighted_swiglu


class Context:
    def __init__(self, path, model, profile='bf16_legacy', *, original=False):
        model.config.save_pretrained(path / 'checkpoint')
        write_new(path / 'mapping.json', {name: name for name, _ in model.named_parameters()})
        self.plan = {'artifacts': {'hf_checkpoint': {'path': str(path / 'checkpoint')},
                                  'hf_parameter_map': {'path': str(path / 'mapping.json')}}}
        options = ({'compute_dtype': 'bfloat16', 'parameter_dtype': 'bfloat16', 'micro_batch_size': 1,
                    'normalization_timing': 'after_backward'} if original else {'profile': profile})
        self.stage = {'configuration': {'options': options, 'device': 'cpu'}}
        self.suite = {'training': {'dtype': 'bfloat16', 'steps': 3, 'global_batch_size': 3, 'clip_grad': 0,
                                  'optimizer': {'name': 'adamw', 'lr': .0007, 'betas': [.8, .95],
                                                'eps': 1e-6, 'weight_decay': .01}},
                      'description': {'parameters': {name: {'shape': list(p.shape)} for name, p in model.named_parameters()},
                                      'inputs': [{'samples': [{'id': i, 'tokens': [1, 2 + i, 1, 4 + s, 7, 1, 8, 9, 10, 11, 12, 13],
                                                               'loss_mask': [0, 1, int(i > 0), 1, int(i > 1), 1, 1, 1, 1, 1, 1, 1]}
                                                              for i in range(3)]} for s in range(3)]}}
        self.metadata, self.steps, self.values, self.sample_contexts = {}, [], {}, []

    def write(self, step, role, name, value):
        key = (step, role, name)
        assert key not in self.values
        self.values[key] = value.detach().clone()

    def observe(self, step, name, value):
        self.write(step, 'observation', name, value)

    def record_step(self, step, **details):
        self.steps.append((step, details))


def tiny_model():
    torch.manual_seed(421)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=12,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                            num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                            rope_theta=1_000_000, attention_dropout=0, tie_word_embeddings=False,
                            router_aux_loss_coef=0, output_router_logits=False)
    config._attn_implementation = 'eager'
    return Qwen3MoeForCausalLM(config)


def loader(monkeypatch, model, ctx=None, fail_forward=False):
    loaded = []

    def load(*args, **kwargs):
        result = deepcopy(model)
        if ctx is not None:
            def observe(module, args, kwargs):
                ctx.sample_contexts.append((ctx.diagnostic_step, ctx.diagnostic_microbatch, list(ctx.diagnostic_sample_ids)))
                if fail_forward:
                    raise RuntimeError('injected MoE forward failure')
            result.register_forward_pre_hook(observe, with_kwargs=True)
        loaded.append(result)
        return result

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', load)
    return loaded


def test_router_fp64_projection_and_bf16_gradient_boundaries():
    torch.manual_seed(31)
    x = torch.randn(7, 8).bfloat16().requires_grad_()
    master = torch.randn(4, 8, requires_grad=True)
    gradient = torch.randn(7, 4)
    output = router_linear(x, master)
    output.backward(gradient)
    xd, wd, gd = x.detach().double(), master.detach().bfloat16().double(), gradient.double()
    torch.testing.assert_close(output, (xd @ wd.T).float(), rtol=0, atol=1e-6)
    assert torch.equal(x.grad, (gd @ wd).bfloat16())
    assert torch.equal(master.grad, (gd.T @ xd).bfloat16().float())
    assert not torch.equal(master.grad, (gd.T @ xd).float())
    # FP32 output carries information that a BF16 router projection discards.
    assert not torch.equal(output, F.linear(x.detach(), master.detach().bfloat16()).float())


def test_weighted_swiglu_fp64_value_and_all_three_derivatives():
    torch.manual_seed(52)
    gate = torch.randn(7, 12).bfloat16().requires_grad_()
    up = torch.randn(7, 12).bfloat16().requires_grad_()
    probabilities = torch.rand(7, 1).requires_grad_()
    gradient = torch.randn(7, 12).bfloat16()
    output = weighted_swiglu(gate, up, probabilities)
    output.backward(gradient)
    x, y, p, dy = [value.detach().double() for value in (gate, up, probabilities, gradient)]
    sigmoid = 1 / (1 + torch.exp(-x))
    assert torch.equal(output, (x * sigmoid * y * p).bfloat16())
    assert torch.equal(gate.grad, (dy * p * y * sigmoid * (1 + x * (1 - sigmoid))).bfloat16())
    assert torch.equal(up.grad, (dy * p * x * sigmoid).bfloat16())
    torch.testing.assert_close(probabilities.grad, (dy * x * sigmoid * y).sum(-1, keepdim=True).float(),
                               rtol=3e-6, atol=1e-6)
    assert not torch.equal(output, F.silu(gate.detach()) * up.detach() * probabilities.detach().bfloat16())


def test_packed_qkv_preserves_weight_meanings_and_one_input_gradient_boundary():
    torch.manual_seed(732)
    attention = tiny_model().model.layers[0].self_attn
    with torch.no_grad():
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            projection.weight.copy_(torch.randn_like(projection.weight))
    x = torch.randn(1, 7, 16).bfloat16().requires_grad_()
    query, key, value = packed_qkv(attention, x)
    gradients = [torch.randn_like(tensor) for tensor in (query, key, value)]
    torch.autograd.backward((query, key, value), gradients)
    desired_dx = torch.zeros_like(x, dtype=torch.float64)
    rounded_parts = []
    for output, gradient, projection in zip((query, key, value), gradients,
                                             (attention.q_proj, attention.k_proj, attention.v_proj)):
        effective = projection.weight.detach().bfloat16()
        direct = F.linear(x.detach(), effective).reshape(output.shape)
        assert torch.equal(output, direct)
        dy = gradient.reshape(-1, effective.shape[0]).double()
        desired_dx += (dy @ effective.double()).reshape(x.shape)
        rounded_parts.append((dy @ effective.double()).bfloat16().reshape(x.shape))
        torch.testing.assert_close(projection.weight.grad,
                                   (dy.T @ x.detach().reshape(-1, 16).double()).float(), rtol=0, atol=2e-6)
    assert torch.equal(x.grad, desired_dx.bfloat16())
    assert not torch.equal(x.grad, rounded_parts[0] + rounded_parts[1] + rounded_parts[2])


def test_legacy_moe_matches_public_bf16_all_evidence_for_three_steps(tmp_path, monkeypatch):
    model = tiny_model()
    original = Context(tmp_path / 'original', model, original=True)
    loader(monkeypatch, model)
    hf.run_reference(original)
    actual = Context(tmp_path / 'actual', model)
    config_before = deepcopy(actual.stage)
    loaded = loader(monkeypatch, model, actual)
    original_loader = AutoModelForCausalLM.__dict__['from_pretrained']
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    run_reference(actual)
    assert actual.values.keys() == original.values.keys()
    for key, value in actual.values.items():
        right = original.values[key]
        assert value.dtype == right.dtype
        assert torch.equal(value.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), key
    assert actual.stage == config_before
    assert actual.sample_contexts == [(s, i, [i]) for s in range(3) for i in range(3)]
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_loader
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert all('forward' not in vars(module) for module in loaded[0].modules())
    assert loaded[0].model.rotary_emb.inv_freq.dtype == torch.float32
    assert not hasattr(actual, 'diagnostic_sample_ids')


@pytest.mark.parametrize('profile', ['router_fp32', 'weighted_experts', 'packed_projections',
                                   'fused_norm_rope_swiglu', 'math_attention', 'all_ops_math'])
def test_profiles_three_steps_routing_audit_and_master_state(tmp_path, monkeypatch, profile):
    model = tiny_model()
    ctx = Context(tmp_path, model, profile)
    loaded = loader(monkeypatch, model, ctx)
    run_reference(ctx)
    meta = ctx.metadata['arithmetic_control']
    assert meta['restored'] and len(meta['steps']) == len(meta['routing_steps']) == len(ctx.steps) == 3
    assert not meta['routing_contract']['forced_routes']
    assert all(torch.isfinite(value).all() for value in ctx.values.values())
    assert all('forward' not in vars(module) for module in loaded[0].modules())
    for step in meta['routing_steps']:
        assert len(step['layers']) == 2
        for name, records in step['layers'].items():
            assert [record['sample_id'] for record in records] == [0, 1, 2]
            for record in records:
                assert record['token_positions'] == list(range(12))
                assert record['next_token_loss_mask'][-1] == 0
                logits = torch.tensor(record['logits'], dtype=torch.float32)
                selected_logits, selected = logits.topk(2, dim=-1)
                assert selected.tolist() == record['selected_experts']
                assert logits.dtype == torch.float32 and record['logit_dtype'] == 'torch.float32'
                assert sha256(logits.numpy().tobytes()).hexdigest() == record['logit_sha256']
                torch.testing.assert_close(torch.tensor(record['probabilities_fp32']), selected_logits.softmax(-1),
                                           rtol=0, atol=0)
                counts = Counter(selected.reshape(-1).tolist())
                assert record['expert_token_counts'] == [counts[index] for index in range(4)]
                assert sum(record['expert_token_counts']) == 24
            summary = meta['steps'][step['step']]['routing'][name]
            assert sum(summary['expert_token_counts']) == 72 and summary['tokens'] == 36
    for name, _ in model.named_parameters():
        assert torch.equal(ctx.values[1, 'initial', name], ctx.values[0, 'parameter', name])
        assert torch.equal(ctx.values[2, 'initial', name], ctx.values[1, 'parameter', name])
        assert torch.equal(ctx.values[2, 'update', name], ctx.values[2, 'parameter', name] - ctx.values[2, 'initial', name])


def test_micro_unused_experts_remain_unused_without_fabricated_gradients(tmp_path, monkeypatch):
    model = tiny_model()
    with torch.no_grad():
        for layer in model.model.layers:
            for parameter in layer.self_attn.parameters():
                if parameter.ndim == 2:
                    parameter.zero_()
            layer.mlp.gate.weight.zero_()
            layer.mlp.gate.weight[:, 0] = torch.tensor([1., .5, -1., -.5])
        model.model.embed_tokens.weight[1, 0] = 1
        model.model.embed_tokens.weight[2, 0] = -1
    ctx = Context(tmp_path, model, 'packed_projections')
    for batch in ctx.suite['description']['inputs']:
        for sample in batch['samples']:
            sample['tokens'] = [1 if sample['id'] != 1 else 2] * 12
    loader(monkeypatch, model)
    run_reference(ctx)
    for step, details in ctx.steps:
        seen = details['details']['seen_micro_count']
        for layer in range(2):
            for expert in range(4):
                for projection in ('gate_proj', 'up_proj', 'down_proj'):
                    assert seen[f'model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'] == (2 if expert < 2 else 1)


def test_length65_router_coverage_and_shifted_mask_include_final_position(tmp_path, monkeypatch):
    model = tiny_model()
    ctx = Context(tmp_path, model, 'math_attention')
    ctx.suite['training']['steps'] = 1
    ctx.suite['description']['inputs'] = ctx.suite['description']['inputs'][:1]
    for sample in ctx.suite['description']['inputs'][0]['samples']:
        sample['tokens'] = [1 + (position + sample['id']) % 29 for position in range(65)]
        sample['loss_mask'] = [0] + [int((position + sample['id']) % 3 != 0) for position in range(1, 65)]
    loader(monkeypatch, model)
    run_reference(ctx)
    assert ctx.values[0, 'observation', 'logits'].shape == (3, 65, 32)
    for records in ctx.metadata['arithmetic_control']['routing_steps'][0]['layers'].values():
        for record, sample in zip(records, ctx.suite['description']['inputs'][0]['samples']):
            assert record['token_positions'] == list(range(65))
            assert len(record['selected_experts']) == 65 and sum(record['expert_token_counts']) == 130
            assert record['next_token_loss_mask'] == sample['loss_mask'][1:] + [0]


def test_whole_step_unused_expert_still_fails_public_seen_rule_and_restores(tmp_path, monkeypatch):
    model = tiny_model()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.zero_()
    ctx = Context(tmp_path, model, 'math_attention')
    loaded = loader(monkeypatch, model)
    original_loader = AutoModelForCausalLM.__dict__['from_pretrained']
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    with pytest.raises(EvidenceError, match='unused trainable parameters'):
        run_reference(ctx)
    assert ctx.steps == []
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_loader
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert all('forward' not in vars(module) for module in loaded[0].modules())
    incomplete = ctx.metadata['arithmetic_control']['incomplete_routing_step']
    assert incomplete['microbatches_seen'] == 3 and not incomplete['completed_optimizer_step']
    assert any(count == 0 for count in next(iter(incomplete['layers'].values()))[0]['expert_token_counts'])


@pytest.mark.parametrize('options', [{'profile': 'unknown'}, {'profile': 'bf16_legacy', 'micro_batch_size': True},
                                   {'profile': 'bf16_legacy', 'parameter_dtype': 'bfloat16'}, {'unknown': 1}])
def test_options_rejected_before_loading(tmp_path, monkeypatch, options):
    model = tiny_model()
    ctx = Context(tmp_path, model)
    ctx.stage['configuration']['options'] = options
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: pytest.fail('must not load'))
    with pytest.raises(EvidenceError):
        run_reference(ctx)


def test_failure_restores_existing_attributes_and_all_globals(tmp_path, monkeypatch):
    model = tiny_model()
    model.model.norm.forward = model.model.norm.forward
    ctx = Context(tmp_path, model, 'math_attention')
    marker = object()
    ctx.diagnostic_sample_ids = marker
    loaded = loader(monkeypatch, model, ctx, fail_forward=True)
    original_loader = AutoModelForCausalLM.__dict__['from_pretrained']
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    with pytest.raises(RuntimeError, match='injected MoE forward failure'):
        run_reference(ctx)
    assert ctx.diagnostic_sample_ids is marker
    assert not hasattr(ctx, 'diagnostic_step') and not hasattr(ctx, 'diagnostic_microbatch')
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_loader
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert 'forward' in vars(loaded[0].model.norm) and ctx.metadata['arithmetic_control']['restored']


@pytest.mark.parametrize('field,value', [('norm_topk_prob', False), ('router_aux_loss_coef', .01),
                                       ('num_experts_per_tok', 1), ('decoder_sparse_step', 2)])
def test_unsupported_semantics_rejected(tmp_path, monkeypatch, field, value):
    model = tiny_model()
    setattr(model.config, field, value)
    ctx = Context(tmp_path, model)
    loader(monkeypatch, model)
    with pytest.raises(EvidenceError, match='configuration'):
        run_reference(ctx)
    assert ctx.metadata['arithmetic_control']['restored']
