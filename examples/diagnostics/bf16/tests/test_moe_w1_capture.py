"""Portable observer checks; no native source checkout or GPU archive is read.

The two reference integrations require Transformers 4.57.6. The remaining
checks use synthetic tensors and do not claim native dispatcher validation.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training_parity.common import EvidenceError, digest
import contract_moe_reference as base
import moe_fa2_reference as fa2
import moe_w1_capture as probe


def reference_dependencies():
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '4.57.6':
        pytest.skip('This reference observer integration requires Transformers 4.57.6')
    from test_contract_moe_reference import loader
    from test_moe_fa2_reference import model_with_head_dim
    return loader, model_with_head_dim


def context(path, model):
    from test_contract_moe_reference import Context
    ctx = Context(path, model, 'moe_fa2')
    ctx.suite['description']['constraints'] = {'num_layers': 2, 'num_experts': 4}
    folder = path / 'evidence'
    rank = folder / 'rank0'
    rank.mkdir(parents=True)
    ctx.writer = SimpleNamespace(folder=folder, rank_dir=rank)
    return ctx


def array(store, descriptor):
    return store.arrays[descriptor['array']]


def test_three_step_reference_capture_preserves_every_canonical_byte_and_actual_scatter(tmp_path, monkeypatch):
    loader, model_with_head_dim = reference_dependencies()
    assert digest(Path(probe.__file__)) == '4678ab70594696be050d872dbdc9b8e2dc7abbd0dbf8636422c2c96ba1c5b5a7'
    model = model_with_head_dim()
    ordinary, observed = context(tmp_path / 'ordinary', model), context(tmp_path / 'observed', model)
    loader(monkeypatch, model)
    fa2.run_reference(ordinary)
    routes, install = base._State.routes, base._State.install_forward
    probe.run_reference(observed)
    assert base._State.routes is routes and base._State.install_forward is install
    assert observed.metadata['moe_w1_hooks']['restored']
    assert observed.values.keys() == ordinary.values.keys()
    for key, actual in observed.values.items():
        expected = ordinary.values[key]
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key
    receipt = observed.metadata['moe_w1_capture']
    assert receipt['records'] == 18 and receipt['baseline_pairing_verified'] is False
    for path, identity in receipt['files'].items():
        target = observed.writer.folder / path
        assert digest(target) == identity['sha256'] and target.stat().st_size == identity['bytes']
    payload = json.loads((observed.writer.rank_dir / 'moe-w1-capture.json').read_text())
    with np.load(observed.writer.rank_dir / 'moe-w1-capture.npz', allow_pickle=False) as data:
        for row in payload['records']:
            get = lambda d: data[d['array']]
            x, selected = get(row['router']['inputs']), get(row['router']['selected'])
            assert row['router']['expert_ids'] == [0, 1, 2, 3]
            assert np.array_equal(get(row['router_projection']['weight']), get(row['router']['weight']))
            pairs = []
            output = torch.zeros_like(torch.from_numpy(x)).bfloat16()
            for expert in row['experts']:
                tokens = get(expert['scatter_index'])
                assert expert['scatter_source_identity_verified']
                assert np.array_equal(get(expert['inputs']), x[tokens])
                assert np.array_equal(get(expert['outputs']), get(expert['scatter_source']))
                assert np.all(np.any(selected[tokens] == expert['expert_id'], axis=1))
                pairs.extend((int(token), expert['expert_id']) for token in tokens)
                output.index_add_(0, torch.from_numpy(tokens), torch.from_numpy(get(expert['outputs'])).bfloat16())
            assert len(pairs) == len(set(pairs)) == len(x) * 2
            assert np.array_equal(output.float().numpy(), get(row['combined_output']))


def test_reference_exception_restores_methods_and_does_not_publish_partial_capture(tmp_path, monkeypatch):
    loader, model_with_head_dim = reference_dependencies()
    model = model_with_head_dim()
    ctx = context(tmp_path, model)
    loader(monkeypatch, model, ctx, fail_forward=True)
    routes, install = base._State.routes, base._State.install_forward
    with pytest.raises(RuntimeError, match='injected MoE forward failure'):
        probe.run_reference(ctx)
    assert base._State.routes is routes and base._State.install_forward is install
    assert ctx.metadata['moe_w1_hooks']['restored']
    assert not list(ctx.writer.rank_dir.iterdir())


def minimal_store(tmp_path):
    ctx = SimpleNamespace(metadata={}, writer=SimpleNamespace(rank_dir=tmp_path, folder=tmp_path),
        suite={'description': {'constraints': {'num_layers': 1, 'num_experts': 4},
              'inputs': [{'samples': [{'id': 9, 'tokens': [1, 2, 3, 4, 5]}]}]}})
    return probe.Store(ctx, 'native')


def test_reference_mode_observes_actual_wrong_index_without_correcting_it(tmp_path):
    store = minimal_store(tmp_path)
    source = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    event, row = {}, {'router': {'observed': True}}
    pending = {id(source): (source, event)}
    output = torch.zeros(5, 2)
    actual_indices = torch.tensor([4, 0])
    with probe.ReferenceMode(store, row, pending):
        returned = output.index_add_(0, actual_indices, source)
    assert returned is output and not pending
    assert array(store, event['scatter_index']).tolist() == [4, 0]
    assert output.tolist() == [[3, 4], [0, 0], [0, 0], [0, 0], [1, 2]]
    output.sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))


def test_reference_mode_rejects_unbound_source_even_equal_values(tmp_path):
    store = minimal_store(tmp_path)
    source = torch.ones(2, 2)
    with pytest.raises(EvidenceError, match='not the captured expert output'):
        with probe.ReferenceMode(store, {'router': {'observed': True}}, {id(source): (source, {})}):
            torch.zeros(5, 2).index_add_(0, torch.tensor([0, 1]), source.clone())


def test_missing_and_duplicate_record_coverage_is_not_published(tmp_path):
    store = minimal_store(tmp_path)
    with pytest.raises(EvidenceError, match='coverage differs'):
        store.finish()
    store.start(0, 1, 9, range(5))
    with pytest.raises(EvidenceError, match='Duplicate'):
        store.start(0, 1, 9, range(5))
    with pytest.raises(EvidenceError, match='coverage differs'):
        store.finish()
    assert not list(tmp_path.iterdir())


def test_dependency_mismatch_fail_closed(tmp_path):
    module = SimpleNamespace(__file__=__file__)
    with pytest.raises(EvidenceError, match='source identity differs'):
        probe.verify_modules({'test': module}, {'test': '0' * 64})
