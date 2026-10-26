"""CPU integration, identity and restoration checks for dense arithmetic profiles."""
from copy import deepcopy

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
import transformers.models.qwen3.modeling_qwen3 as qwen

from training_parity.common import EvidenceError, write_new
from slime_parity.adapters import hf
from contract_reference import run_reference


class Context:
    def __init__(self, path, model, profile='bf16_legacy', *, original=False):
        model.config.save_pretrained(path / 'checkpoint')
        write_new(path / 'mapping.json', {n: n for n, _ in model.named_parameters()})
        self.plan = {'artifacts': {'hf_checkpoint': {'path': str(path / 'checkpoint')},
                                  'hf_parameter_map': {'path': str(path / 'mapping.json')}}}
        options = ({'compute_dtype': 'bfloat16', 'parameter_dtype': 'bfloat16', 'micro_batch_size': 1,
                    'normalization_timing': 'after_backward'} if original else {'profile': profile})
        self.stage = {'configuration': {'options': options, 'device': 'cpu'}}
        self.suite = {'training': {'dtype': 'bfloat16', 'steps': 3, 'global_batch_size': 3, 'clip_grad': 0,
                                  'optimizer': {'name': 'adamw', 'lr': .0007, 'betas': [.8, .95],
                                                'eps': 1e-6, 'weight_decay': .01}},
                      'description': {'parameters': {n: {'shape': list(p.shape)} for n, p in model.named_parameters()},
                                      'inputs': [{'samples': [{'id': i, 'tokens': [1, 2 + i, 1, 4 + s, 7, 1],
                                                               'loss_mask': [0, 1, int(i > 0), 1, int(i > 1), 1]}
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
    torch.manual_seed(98)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         rope_theta=1_000_000, attention_dropout=0, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config)


def loader(monkeypatch, model, ctx=None, fail_forward=False):
    loaded = []

    def load(*args, **kwargs):
        result = deepcopy(model)
        if ctx is not None:
            def observe(module, args, kwargs):
                ctx.sample_contexts.append((ctx.diagnostic_step, ctx.diagnostic_microbatch, list(ctx.diagnostic_sample_ids)))
                if fail_forward:
                    raise RuntimeError('injected forward failure')
            result.register_forward_pre_hook(observe, with_kwargs=True)
        loaded.append(result)
        return result

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', load)
    return loaded


def assert_same_bytes(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        right = expected[key]
        assert value.dtype == right.dtype
        assert torch.equal(value.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), key


def test_legacy_matches_original_bf16_hf_three_steps_and_preserves_context(tmp_path, monkeypatch):
    model = tiny_model()
    expected = Context(tmp_path / 'original', model, original=True)
    loader(monkeypatch, model)
    hf.run_reference(expected)
    actual = Context(tmp_path / 'legacy', model)
    initial_stage = deepcopy(actual.stage)
    loaded = loader(monkeypatch, model, actual)
    original_load = AutoModelForCausalLM.__dict__['from_pretrained']
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    run_reference(actual)
    assert_same_bytes(actual.values, expected.values)
    assert actual.stage == initial_stage
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_load
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert actual.metadata['arithmetic_control']['restored']
    assert all('forward' not in vars(module) for module in loaded[0].modules())
    assert all(p.dtype == torch.float32 for p in loaded[0].parameters())
    assert actual.sample_contexts == [(s, i, [i]) for s in range(3) for i in range(3)]
    assert not any(hasattr(actual, name) for name in ('diagnostic_sample_ids', 'diagnostic_step', 'diagnostic_microbatch'))
    assert loaded[0].model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(loaded[0].model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq)


@pytest.mark.parametrize('profile', ['linear_fp32wgrad', 'fused_norm_rope_swiglu', 'math_attention', 'all_ops_math'])
def test_profiles_complete_three_steps_with_declared_operator_coverage(tmp_path, monkeypatch, profile):
    model = tiny_model()
    ctx = Context(tmp_path, model, profile)
    loaded = loader(monkeypatch, model, ctx)
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    run_reference(ctx)
    meta = ctx.metadata['arithmetic_control']
    assert len(ctx.steps) == len(meta['steps']) == 3 and meta['restored']
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert all('forward' not in vars(module) for module in loaded[0].modules())
    assert all(torch.isfinite(value).all() for value in ctx.values.values())
    assert all(value.dtype == torch.float32 for (step, role, name), value in ctx.values.items() if role != 'observation')
    assert all(ctx.values[s, 'observation', 'logits'].dtype == torch.bfloat16 for s in range(3))
    for step in meta['steps']:
        calls = step['operator_calls']
        assert step['sample_ids'] == [0, 1, 2]
        assert calls['model_forward'] == calls['model.embed_tokens'] == calls['model.rotary_emb'] == 3
        assert calls['rope_apply'] == 6
        if profile in ('math_attention', 'all_ops_math'):
            assert calls['model.layers.0.self_attn.math_attention'] == 3
            assert meta['module_strategies']['model.embed_tokens'] == 'chunk32_bf16_gradient'
    # The unrounded master at step N is exactly the next step's initial value.
    for name, _ in model.named_parameters():
        assert torch.equal(ctx.values[1, 'initial', name], ctx.values[0, 'parameter', name])
        assert torch.equal(ctx.values[2, 'initial', name], ctx.values[1, 'parameter', name])
        assert torch.equal(ctx.values[2, 'update', name], ctx.values[2, 'parameter', name] - ctx.values[2, 'initial', name])


def test_linear_only_preserves_first_forward_and_changes_only_linear_weight_gradients(tmp_path, monkeypatch):
    model = tiny_model()
    legacy = Context(tmp_path / 'legacy', model)
    linear = Context(tmp_path / 'linear', model, 'linear_fp32wgrad')
    loader(monkeypatch, model)
    run_reference(legacy)
    run_reference(linear)
    assert torch.equal(linear.values[0, 'observation', 'logits'], legacy.values[0, 'observation', 'logits'])
    assert torch.equal(linear.values[0, 'observation', 'loss'], legacy.values[0, 'observation', 'loss'])
    linear_names = {name + '.weight' for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    changed = []
    for name, _ in model.named_parameters():
        if not torch.equal(linear.values[0, 'gradient', name], legacy.values[0, 'gradient', name]):
            changed.append(name)
    assert changed and set(changed) <= linear_names


def test_forward_failure_restores_existing_instance_attribute_and_globals(tmp_path, monkeypatch):
    model = tiny_model()
    # Preserve a caller's instance-level forward override, not merely class behavior.
    original_instance_forward = model.model.norm.forward
    model.model.norm.forward = original_instance_forward
    ctx = Context(tmp_path, model, 'math_attention')
    marker = object()
    ctx.diagnostic_sample_ids = marker
    loaded = loader(monkeypatch, model, ctx, fail_forward=True)
    original_load = AutoModelForCausalLM.__dict__['from_pretrained']
    original_rope, original_attention = qwen.apply_rotary_pos_emb, qwen.eager_attention_forward
    with pytest.raises(RuntimeError, match='injected forward failure'):
        run_reference(ctx)
    assert ctx.diagnostic_sample_ids is marker
    assert not hasattr(ctx, 'diagnostic_step') and not hasattr(ctx, 'diagnostic_microbatch')
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_load
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_attention
    assert 'forward' in vars(loaded[0].model.norm)
    assert loaded[0].model.norm.forward.__func__ is original_instance_forward.__func__
    assert ctx.metadata['arithmetic_control']['restored'] and ctx.steps == []


@pytest.mark.parametrize('options', [{'profile': 'unknown'}, {'profile': 'bf16_legacy', 'micro_batch_size': 2},
                                   {'profile': 'bf16_legacy', 'micro_batch_size': True},
                                   {'profile': 'bf16_legacy', 'micro_batch_size': 1.0},
                                   {'profile': 'bf16_legacy', 'parameter_dtype': 'bfloat16'}, {'unknown': 1}])
def test_invalid_options_rejected_before_model_loading(tmp_path, monkeypatch, options):
    model = tiny_model()
    ctx = Context(tmp_path, model)
    ctx.stage['configuration']['options'] = options
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: pytest.fail('must not load'))
    with pytest.raises(EvidenceError):
        run_reference(ctx)


def test_shared_weights_rejected_and_loader_restored(tmp_path, monkeypatch):
    model = tiny_model()
    model.lm_head.weight = model.model.embed_tokens.weight
    ctx = Context(tmp_path, model)
    loader(monkeypatch, model)
    original = AutoModelForCausalLM.__dict__['from_pretrained']
    with pytest.raises(EvidenceError, match='Shared parameters'):
        run_reference(ctx)
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original
    assert ctx.metadata['arithmetic_control']['restored']
