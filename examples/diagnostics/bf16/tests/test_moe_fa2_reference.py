"""MoE FA2 overlay integration, scope and restoration checks on CPU."""
from copy import deepcopy

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen

from training_parity.common import EvidenceError
import bf16_ops as ops
import contract_moe_reference as base
import moe_fa2_reference as overlay
from test_contract_moe_reference import Context, loader


def model_with_head_dim(width=32):
    torch.manual_seed(765)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=2 * width, intermediate_size=24, moe_intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=width,
                            num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                            rope_theta=1_000_000, attention_dropout=0, tie_word_embeddings=False,
                            router_aux_loss_coef=0, output_router_logits=False)
    config._attn_implementation = 'eager'
    return Qwen3MoeForCausalLM(config)


def snapshot(ctx):
    return (ctx.stage, deepcopy(ctx.stage), base._install, ops.attention_fp32,
            AutoModelForCausalLM.__dict__['from_pretrained'], qwen.apply_rotary_pos_emb, qwen.eager_attention_forward)


def assert_restored(ctx, before, model):
    stage, stage_copy, install, attention, load, rope, eager = before
    assert ctx.stage is stage and ctx.stage == stage_copy
    assert base._install is install and ops.attention_fp32 is attention
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is load
    assert qwen.apply_rotary_pos_emb is rope and qwen.eager_attention_forward is eager
    assert ctx.metadata['moe_fa2_overlay']['restored']
    assert ctx.metadata['arithmetic_control']['restored']
    assert all('forward' not in vars(module) for module in model.modules())


def test_three_steps_use_fa2_and_restore_context_with_sample_ids_visible(tmp_path, monkeypatch):
    model = model_with_head_dim()
    ctx = Context(tmp_path, model, overlay.PROFILE)
    loaded = loader(monkeypatch, model, ctx)
    before = snapshot(ctx)
    overlay.run_reference(ctx)
    assert_restored(ctx, before, loaded[0])
    assert ctx.sample_contexts == [(step, micro, [micro]) for step in range(3) for micro in range(3)]
    assert not any(hasattr(ctx, name) for name in ('diagnostic_step', 'diagnostic_sample_ids', 'diagnostic_microbatch'))
    meta = ctx.metadata['arithmetic_control']
    assert meta['profile'] == meta['requested_profile'] == overlay.PROFILE
    assert meta['attention'] == ctx.metadata['reference_precision']['attention'] == overlay.ATTENTION_LABEL
    assert ctx.metadata['moe_fa2_overlay']['attention_forward_calls'] == 18
    assert len(meta['steps']) == len(meta['routing_steps']) == len(ctx.steps) == 3
    assert all(record['profile'] == overlay.PROFILE for record in meta['steps'])
    assert all(record[1]['details']['arithmetic_control']['profile'] == overlay.PROFILE for record in ctx.steps)
    assert all(strategy == 'independent_one_tile_fa2_source_boundaries'
               for name, strategy in meta['module_strategies'].items() if name.endswith('.math_attention'))
    assert all(torch.isfinite(value).all() for value in ctx.values.values())
    for name, _ in model.named_parameters():
        assert torch.equal(ctx.values[1, 'initial', name], ctx.values[0, 'parameter', name])
        assert torch.equal(ctx.values[2, 'initial', name], ctx.values[1, 'parameter', name])


@pytest.mark.parametrize('length,width,error', [(129, 32, 'length <= 128'), (12, 16, 'D32')])
def test_scope_rejection_restores_all_temporary_changes(tmp_path, monkeypatch, length, width, error):
    model = model_with_head_dim(width)
    ctx = Context(tmp_path, model, overlay.PROFILE)
    for batch in ctx.suite['description']['inputs']:
        for sample in batch['samples']:
            sample['tokens'] = [1 + position % 29 for position in range(length)]
            sample['loss_mask'] = [0] + [1] * (length - 1)
    loaded = loader(monkeypatch, model)
    before = snapshot(ctx)
    with pytest.raises(EvidenceError, match=error):
        overlay.run_reference(ctx)
    assert_restored(ctx, before, loaded[0])
    assert ctx.steps == [] and ctx.metadata['moe_fa2_overlay']['attention_forward_calls'] == 0


def test_forward_exception_preserves_existing_sample_attribute_and_restores(tmp_path, monkeypatch):
    model = model_with_head_dim()
    ctx = Context(tmp_path, model, overlay.PROFILE)
    marker = object()
    ctx.diagnostic_sample_ids = marker
    loaded = loader(monkeypatch, model, ctx, fail_forward=True)
    before = snapshot(ctx)
    with pytest.raises(RuntimeError, match='injected MoE forward failure'):
        overlay.run_reference(ctx)
    assert_restored(ctx, before, loaded[0])
    assert ctx.diagnostic_sample_ids is marker
    assert not hasattr(ctx, 'diagnostic_step') and not hasattr(ctx, 'diagnostic_microbatch')


def test_dependency_mismatch_rejected_before_loading_or_patching(tmp_path, monkeypatch):
    model = model_with_head_dim()
    ctx = Context(tmp_path, model, overlay.PROFILE)
    loaded = loader(monkeypatch, model)
    before = snapshot(ctx)
    monkeypatch.setitem(overlay.DEPENDENCIES, 'attention_contract_probe.py', '0' * 64)
    with pytest.raises(EvidenceError, match='dependency changed'):
        overlay.run_reference(ctx)
    assert loaded == [] and ctx.metadata == {} and ctx.stage is before[0]
    assert base._install is before[2] and ops.attention_fp32 is before[3]


@pytest.mark.parametrize('options', [{}, {'profile': 'math_attention'}, {'profile': 'moe_fa2', 'forced_routes': []}])
def test_invalid_overlay_options_rejected_without_patching(tmp_path, monkeypatch, options):
    model = model_with_head_dim()
    ctx = Context(tmp_path, model, overlay.PROFILE)
    ctx.stage['configuration']['options'] = options
    loaded = loader(monkeypatch, model)
    before = snapshot(ctx)
    with pytest.raises(EvidenceError, match='option'):
        overlay.run_reference(ctx)
    assert loaded == [] and ctx.metadata == {} and ctx.stage is before[0]
    assert base._install is before[2] and ops.attention_fp32 is before[3]
