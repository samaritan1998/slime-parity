"""CPU identity and restoration checks for the altered Qwen3 RoPE diagnostic."""
from copy import deepcopy
from importlib import metadata, util
from pathlib import Path

import pytest
import torch

from training_parity.common import write_new
from slime_parity.adapters import hf

_DIAGNOSTIC_PATH = Path(__file__).resolve().parents[1] / 'examples/diagnostics/qwen3_rope_reference.py'
_DIAGNOSTIC_SPEC = util.spec_from_file_location('qwen3_rope_reference_test_example', _DIAGNOSTIC_PATH)
_DIAGNOSTIC = util.module_from_spec(_DIAGNOSTIC_SPEC)
_DIAGNOSTIC_SPEC.loader.exec_module(_DIAGNOSTIC)
EXPECTED_OPTIONS = _DIAGNOSTIC.EXPECTED_OPTIONS
_apply_rotary_fp32 = _DIAGNOSTIC._apply_rotary_fp32
_fp32_rope = _DIAGNOSTIC._fp32_rope
run_reference = _DIAGNOSTIC.run_reference


@pytest.fixture
def pinned_hf():
    try:
        installed = metadata.version('transformers')
    except metadata.PackageNotFoundError:
        pytest.skip('Qwen3 RoPE integration requires transformers==4.57.6; dependency is absent')
    if installed != '4.57.6':
        pytest.skip(f'Qwen3 RoPE integration requires transformers==4.57.6; found {installed}')
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    import transformers.models.qwen3.modeling_qwen3 as qwen

    return AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, qwen


@pytest.mark.parametrize('layout', ['bhld', 'blhd'])
def test_rotation_positions_layout_dtype_and_transpose_gradient(layout):
    torch.manual_seed(47)
    positions = torch.tensor([[0, 1, 7, 31, 63], [0, 2, 8, 32, 64]])
    inverse = 1.0 / (1_000_000 ** (torch.arange(0, 8, 2).float() / 8))
    angles = positions.float()[..., None] * inverse
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos(), angles.sin()
    query = torch.randn(2, 4, 5, 8).bfloat16()
    key = torch.randn(2, 2, 5, 8).bfloat16()
    axis = 1
    if layout == 'blhd':
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        axis = 2
    query.requires_grad_()
    key.requires_grad_()
    qo, ko = _apply_rotary_fp32(query, key, cos, sin, unsqueeze_dim=axis)
    for source, result in ((query, qo), (key, ko)):
        assert result.dtype == torch.bfloat16 and result.shape == source.shape
        # Explicit two-by-two rotation matrix rather than rotate_half helper.
        first, second = source.double().chunk(2, dim=-1)
        c, s = cos[..., :4].double().unsqueeze(axis), sin[..., :4].double().unsqueeze(axis)
        expected = torch.cat((first * c - second * s, first * s + second * c), -1).bfloat16()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        token_dim = 2 if axis == 1 else 1
        assert torch.equal(result.select(token_dim, 0), source.select(token_dim, 0))
        upstream = (torch.arange(result.numel()).reshape(result.shape) % 7 - 3).bfloat16()
        result.backward(upstream)
        u, v = upstream.double().chunk(2, -1)
        transpose_rotation = torch.cat((u * c + v * s, -u * s + v * c), -1).bfloat16()
        torch.testing.assert_close(source.grad, transpose_rotation, atol=0, rtol=0)


@pytest.mark.parametrize('fail', [False, True])
def test_scope_preserves_fp32_identity_and_restores_attributes(fail, pinned_hf):
    _, Qwen3Config, _, qwen = pinned_hf
    config = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_theta=1_000_000, attention_dropout=0)
    rotary = qwen.Qwen3RotaryEmbedding(config)
    positions = torch.tensor([[0, 1, 7, 31, 63]])
    x = torch.randn(1, 5, 32)
    q, k = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8)
    original_forward = qwen.Qwen3RotaryEmbedding.__dict__['forward']
    original_apply = qwen.apply_rotary_pos_emb
    buffers = {name: value.clone() for name, value in rotary.named_buffers()}
    cosine, sine = rotary(x, positions)
    expected = original_apply(q, k, cosine, sine)
    counts = {'coefficient_calls': 0, 'rotation_calls': 0}
    try:
        with _fp32_rope(qwen, counts):
            actual_cos, actual_sin = rotary(x.bfloat16(), positions)
            assert torch.equal(cosine, actual_cos) and torch.equal(sine, actual_sin)
            actual = qwen.apply_rotary_pos_emb(q, k, actual_cos, actual_sin)
            for a, b in zip(actual, expected):
                assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
            if fail:
                raise RuntimeError('intentional failure')
    except RuntimeError as error:
        assert fail and str(error) == 'intentional failure'
    assert qwen.Qwen3RotaryEmbedding.__dict__['forward'] is original_forward
    assert qwen.apply_rotary_pos_emb is original_apply
    assert counts == {'coefficient_calls': 1, 'rotation_calls': 1, 'attributes_restored': True}
    for name, value in rotary.named_buffers():
        assert value.dtype == buffers[name].dtype and torch.equal(value, buffers[name])


class Context:
    def __init__(self, path, model):
        model.config.save_pretrained(path / 'checkpoint')
        write_new(path / 'mapping.json', {name: name for name, _ in model.named_parameters()})
        self.plan = {'artifacts': {'hf_checkpoint': {'path': str(path / 'checkpoint')},
                                  'hf_parameter_map': {'path': str(path / 'mapping.json')}}}
        self.stage = {'configuration': {'options': dict(EXPECTED_OPTIONS), 'device': 'cpu'}}
        self.suite = {'training': {'dtype': 'bfloat16', 'steps': 3, 'global_batch_size': 2,
                                  'clip_grad': 0, 'optimizer': {'name': 'adamw', 'lr': .001,
                                  'betas': [.9, .99], 'eps': 1e-8, 'weight_decay': 0}},
                      'description': {'parameters': {n: {'shape': list(p.shape)} for n, p in model.named_parameters()},
                                      'inputs': [{'samples': [{'id': i, 'tokens': [1, 3 + i, 4 + step, 7],
                                                               'loss_mask': [0, 1, 1, 1]} for i in range(2)]}
                                                 for step in range(3)]}}
        self.metadata, self.steps, self.values = {}, [], {}

    def write(self, step, role, name, value):
        self.values[step, role, name] = value.detach().clone()

    def observe(self, step, name, value):
        self.write(step, 'observation', name, value)

    def record_step(self, step, **details):
        self.steps.append((step, details))


def test_pinned_public_runner_three_steps_and_global_restoration(tmp_path, monkeypatch, pinned_hf):
    AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, qwen = pinned_hf
    torch.manual_seed(93)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         rope_theta=1_000_000, attention_dropout=0, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config)
    ctx = Context(tmp_path / 'diagnostic', model)
    initial_parameters = {name: p.detach().bfloat16().float() for name, p in model.named_parameters()}
    original_forward = qwen.Qwen3RotaryEmbedding.__dict__['forward']
    original_apply = qwen.apply_rotary_pos_emb
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *args, **kwargs: deepcopy(model))
    expected = Context(tmp_path / 'expected', model)
    hf.run_reference(expected)
    run_reference(ctx)
    assert len(ctx.steps) == 3
    assert qwen.Qwen3RotaryEmbedding.__dict__['forward'] is original_forward
    assert qwen.apply_rotary_pos_emb is original_apply
    diagnostic = ctx.metadata['rope_arithmetic_diagnostic']
    assert diagnostic['counts'] == {'coefficient_calls': 6, 'rotation_calls': 6, 'attributes_restored': True}
    assert 'not a replacement acceptance oracle' in diagnostic['purpose']
    for name, parameter in initial_parameters.items():
        assert torch.equal(ctx.values[0, 'initial', name], parameter)
    assert all(torch.isfinite(value).all() for value in ctx.values.values())
    # A later ordinary reference must be byte-identical after restoration.
    after = Context(tmp_path / 'after', model)
    hf.run_reference(after)
    assert after.values.keys() == expected.values.keys()
    for key, value in after.values.items():
        assert torch.equal(value.reshape(-1).view(torch.uint8), expected.values[key].reshape(-1).view(torch.uint8)), key


def test_runner_exception_restores_class_and_module_functions(tmp_path, monkeypatch, pinned_hf):
    AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, qwen = pinned_hf
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0)
    ctx = Context(tmp_path, Qwen3ForCausalLM(config))
    original_forward = qwen.Qwen3RotaryEmbedding.__dict__['forward']
    original_apply = qwen.apply_rotary_pos_emb

    def fail(*args, **kwargs):
        raise RuntimeError('checkpoint failure')

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', fail)
    with pytest.raises(RuntimeError, match='checkpoint failure'):
        run_reference(ctx)
    assert qwen.Qwen3RotaryEmbedding.__dict__['forward'] is original_forward
    assert qwen.apply_rotary_pos_emb is original_apply
    assert ctx.metadata['rope_arithmetic_diagnostic']['counts']['attributes_restored'] is True
