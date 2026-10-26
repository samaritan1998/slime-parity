"""Generation-only checks; these temporary seeds are not evaluation fixtures."""
from copy import deepcopy
import random

import pytest
import torch
from transformers import Qwen3ForCausalLM, Qwen3MoeForCausalLM

from training_parity.common import EvidenceError, digest, object_hash, read, write_new
from seeded_fixture import generate, make_cases


def forbidden_forward(*args, **kwargs):
    pytest.fail('Fixture generation must never inspect model outputs/routing')


def test_seeds_are_independent_reproducible_and_do_not_change_global_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(Qwen3ForCausalLM, 'forward', forbidden_forward)
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
    a = generate(tmp_path / 'a', kind='dense', weight_seed=11, token_seed=31, steps=2, global_batch_size=3)
    b = generate(tmp_path / 'b', kind='dense', weight_seed=11, token_seed=32, steps=2, global_batch_size=3)
    c = generate(tmp_path / 'c', kind='dense', weight_seed=12, token_seed=31, steps=2, global_batch_size=3)
    again = generate(tmp_path / 'again', kind='dense', weight_seed=11, token_seed=31, steps=2, global_batch_size=3)
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state
    assert a['parameters'] == b['parameters'] == again['parameters']
    assert a['parameters'] != c['parameters'] and a['cases'] == c['cases'] == again['cases']
    assert a['cases'] != b['cases']
    assert a == again
    assert a['files']['checkpoint/model.safetensors'] == b['files']['checkpoint/model.safetensors']
    for manifest, destination in ((a, tmp_path / 'a'), (b, tmp_path / 'b'), (c, tmp_path / 'c')):
        assert not manifest['evaluation_performed'] and not manifest['routing_inspected']
        for name, receipt in manifest['files'].items():
            assert digest(destination / name) == receipt['sha256']
            assert (destination / name).stat().st_size == receipt['bytes']


def test_moe_generation_has_no_forward_and_keeps_buffers_fp32(tmp_path, monkeypatch):
    monkeypatch.setattr(Qwen3MoeForCausalLM, 'forward', forbidden_forward)
    result = generate(tmp_path / 'moe', kind='moe', weight_seed=13, token_seed=33, steps=1, global_batch_size=2)
    assert result['status'] == 'generated_only'
    assert sum('.mlp.experts.' in name for name in result['parameters']) == 4 * 4 * 3
    assert all(buffer['dtype'] == 'torch.float32' for buffer in result['buffers'].values())
    assert result['cases']['text']['sequence_length'] == 64
    assert result['cases']['masked-boundaries']['sequence_length'] == 65


def test_template_keeps_policy_training_matrix_and_masks_unchanged(tmp_path):
    source = tmp_path / 'source'
    first = generate(source, kind='dense', weight_seed=14, token_seed=34, steps=2, global_batch_size=3)
    mapping = source / 'identity-map.json'
    write_new(mapping, {name: name for name in first['parameters']})
    spec = {'name': 'unit-only',
            'model': {'adapter': 'slime_parity.adapters.hf:HFCausalLMAdapter',
                      'options': {'checkpoint': str(source / 'checkpoint'), 'parameter_map': str(mapping)}},
            'candidate': {'options': {'args': ['--hf-checkpoint', str(source / 'checkpoint'), '--other', 'constant']}},
            'reference': {'runner': 'unexecuted.unit:reference', 'options': {}},
            'training': {'seed': 417, 'steps': 2, 'global_batch_size': 3, 'dtype': 'bfloat16'},
            'cases': read(source / 'cases.json'), 'matrix': {'world_size': [1, 8]},
            'policy': [{'role': 'gradient', 'max_abs': .0001}], 'execution': {'timeout_seconds': 123}}
    template = source / 'template.json'
    write_new(template, spec)
    frozen = deepcopy(spec)
    result = generate(tmp_path / 'target', kind='dense', weight_seed=15, token_seed=35,
                      steps=2, global_batch_size=3, template_spec=template)
    changed = read(tmp_path / 'target/spec.json')
    assert read(template) == frozen
    for key in ('policy', 'training', 'matrix', 'execution', 'reference'):
        assert changed[key] == frozen[key]
    assert result['template']['policy_sha256'] == result['template']['source_policy_sha256'] == object_hash(spec['policy'])
    assert result['template']['runtime_training_seed'] == 417
    assert changed['candidate']['options']['args'] == ['--hf-checkpoint', str(tmp_path / 'target/checkpoint'), '--other', 'constant']
    for before, after in zip(frozen['cases'], changed['cases']):
        assert before['id'] == after['id']
        for old_batch, new_batch in zip(before['inputs'], after['inputs']):
            for old, new in zip(old_batch['samples'], new_batch['samples']):
                assert old['loss_mask'] == new['loss_mask'] and len(old['tokens']) == len(new['tokens'])
                assert old['tokens'] != new['tokens']
    incompatible = deepcopy(frozen)
    incompatible['training']['steps'] = 3
    bad_template = source / 'bad-template.json'
    write_new(bad_template, incompatible)
    with pytest.raises(EvidenceError, match='training step/global batch'):
        generate(tmp_path / 'bad-count', kind='dense', weight_seed=15, token_seed=35,
                 steps=2, global_batch_size=3, template_spec=bad_template)
    assert not (tmp_path / 'bad-count').exists()


def test_existing_destination_refused_without_changes(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_bytes(b'owned elsewhere')
    with pytest.raises(FileExistsError):
        generate(output, kind='dense', weight_seed=16, token_seed=36)
    assert sentinel.read_bytes() == b'owned elsewhere' and list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize('weight,token', [(True, 31), (-1, 31), (11, True), (11, -1), (2 ** 63, 31)])
def test_invalid_seeds_rejected_without_output(tmp_path, weight, token):
    destination = tmp_path / 'invalid'
    with pytest.raises(EvidenceError):
        generate(destination, kind='dense', weight_seed=weight, token_seed=token)
    assert not destination.exists()


def test_mask_boundaries_and_rng_stream_are_fixed_independent_of_weights():
    cases = make_cases(17, steps=2, global_batch_size=3)
    rng = random.Random(17)
    assert cases[0]['inputs'][0]['samples'][0]['tokens'] == [rng.randrange(1, 512) for _ in range(64)]
    for case in cases:
        for batch in case['inputs']:
            for sample in batch['samples']:
                mask = sample['loss_mask']
                assert mask[0] == 0 and sum(mask) > 0 and len(mask) == len(sample['tokens'])
                if case['id'] == 'masked-boundaries':
                    assert mask == [0] + [int((position + sample['id']) % 3 != 0) for position in range(1, 65)]
