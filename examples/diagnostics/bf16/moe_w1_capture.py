"""Read-only W1 MoE routing, actual dispatch and combine observations.

This observer never supplies expected routes or replaces an arithmetic result.
It records actual call arguments/results. Postprocessing must authenticate the
rank artifacts and require all 1428 canonical tensor/steps byte-identical to
each endpoint's uninstrumented baseline before using this evidence. Capturing
already weighted expert outputs does not prove internal probability weighting.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from training_parity.common import digest, require


REFERENCE_PINS = {
    'contract_moe_reference': '21a7acbaf343e371763dfbe8806c1ad58c591b8162a55d1544c3c55c7129145e',
    'moe_fa2_reference': 'be2cdc08870110a880ac68f7932e834a86b3ef28cbc87e434c74144f701c88fc',
}
NATIVE_PINS = {
    'token_dispatcher': '7e20f0d82d983f6ae3b6861fc72eb1b58d9c4ad45549871571627826d9931de7',
    'moe_utils': '700ce2f9c77c6c51ee5806d366be7ad20beda98014278d0cc9aeb22d766d05ca',
    'router': '13e585bbe4021fc72b183508d4aae6d6b41901f61186eff3533d21412aa40497',
    'experts': 'e9973a2905c2f468eb79164694e6cdd63b6453d386cbbbc0264175352d315469',
    'moe_layer': 'b14caab25a724e5496a5aad98da6cea479c051fac10ee4d4e4231a54c460ac49',
}


def runtime_flags():
    return {'torch_version': str(torch.__version__),
            'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'allow_bf16_reduced_precision_reduction': bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_autocast_enabled': torch.is_autocast_enabled('cuda'),
            'cpu_autocast_enabled': torch.is_autocast_enabled('cpu'),
            'projection_instruction_sequence_observed': False}


def function_identity(function):
    result = {'module': getattr(function, '__module__', None),
              'qualname': getattr(function, '__qualname__', type(function).__qualname__)}
    try:
        source = inspect.getsourcefile(function)
    except (TypeError, OSError):
        source = None
    result['source_file_sha256'] = digest(Path(source)) if source and Path(source).is_file() else None
    return result


def verify_modules(modules, pins):
    identities = {}
    for name, expected in pins.items():
        actual = digest(Path(modules[name].__file__))
        require(actual == expected, 'W1 capture source identity differs: ' + name)
        identities[name] = actual
    return identities


class Restore:
    def __init__(self):
        self.items = []

    def replace(self, owner, name, value):
        self.items.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)

    def close(self):
        for owner, name, existed, value in reversed(self.items):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.items.clear()


class Store:
    def __init__(self, ctx, side):
        self.ctx, self.side = ctx, side
        self.records, self.arrays, self.seen = [], {}, set()
        self.sources = {}

    def tensor(self, value):
        require(isinstance(value, torch.Tensor), 'Observation requires an actual tensor')
        dtype, shape = str(value.dtype), list(value.shape)
        with torch.no_grad():
            data = value.detach()
            if data.dtype == torch.bfloat16:
                data = data.float()
            array = data.cpu().numpy().copy()
        require(array.dtype.kind != 'O', 'Object arrays are not evidence')
        name = 'a' + str(len(self.arrays))
        self.arrays[name] = array
        return {'array': name, 'actual_dtype': dtype, 'shape': shape,
                'storage_dtype': str(array.dtype)}

    def start(self, step, layer, sample_id, positions):
        key = int(step), int(layer), int(sample_id)
        require(key not in self.seen, 'Duplicate W1 route/dispatch invocation')
        self.seen.add(key)
        return {'step': key[0], 'layer': key[1], 'sample_id': key[2],
                'positions': [int(i) for i in positions], 'runtime': runtime_flags(),
                'router': {}, 'helpers': [], 'stages': [], 'experts': []}

    def commit(self, row):
        require(row['router'] and 'combined_output' in row, 'Incomplete route/dispatch observation')
        if self.side == 'native':
            kinds = [(r['kind'], r['phase']) for r in row['helpers']]
            require(kinds == [('permute', 'dispatch_preprocess'),
                              ('sort_chunks_by_idxs', 'dispatch_postprocess'),
                              ('sort_chunks_by_idxs', 'combine_preprocess'),
                              ('unpermute', 'combine_postprocess')],
                    'Native dispatcher helper coverage differs')
            require(len(row['experts']) == 1 and len(row['stages']) == 6,
                    'Native expert/stage invocation coverage differs')
        else:
            require(all('scatter_index' in item and item['scatter_source_identity_verified']
                        for item in row['experts']), 'Reference expert/scatter binding missing')
        self.records.append(row)

    def finish(self):
        batches = self.ctx.suite['description']['inputs']
        constraints = self.ctx.suite['description']['constraints']
        expected = {(step, layer, int(sample['id'])) for step, batch in enumerate(batches)
                    for layer in range(1, constraints['num_layers'] + 1) for sample in batch['samples']}
        require(self.seen == expected and len(self.records) == len(expected),
                'W1 capture full step/layer/sample coverage differs')
        for row in self.records:
            sample = next(s for s in batches[row['step']]['samples'] if s['id'] == row['sample_id'])
            require(row['positions'] == list(range(len(sample['tokens']))),
                    'W1 capture requires the actual unpadded one-sample token domain')
            require(row['router']['inputs']['shape'][0] == len(row['positions']),
                    'Router input/token coordinate count differs')
        folder = self.ctx.writer.rank_dir
        manifest, archive = folder / 'moe-w1-capture.json', folder / 'moe-w1-capture.npz'
        require(not manifest.exists() and not archive.exists(), 'W1 capture artifacts already exist')
        payload = {'schema_version': 1, 'side': self.side, 'sources': self.sources,
                   'coverage': {'steps': len(batches), 'layers': constraints['num_layers'],
                                'records': len(self.records), 'record_coverage_complete': True},
                   'baseline_pairing_verified': False, 'required_paired_canonical_tensor_steps': 1428,
                   'internal_probability_weighting_observed': False,
                   'scope': 'W1 actual call boundaries; no drop/pad/fusion or forced routes; no acceptance conclusion',
                   'records': self.records}
        np.savez(archive, **self.arrays)
        manifest.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        self.ctx.metadata['moe_w1_capture'] = {
            'schema_version': 1, 'side': self.side, 'records': len(self.records),
            'record_coverage_complete': True, 'baseline_pairing_verified': False,
            'required_paired_canonical_tensor_steps': 1428,
            'files': {str(path.relative_to(self.ctx.writer.folder)):
                      {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in (manifest, archive)}}


class ReferenceMode(TorchFunctionMode):
    """Observe the original scatter call, never recompute its desired indices."""
    def __init__(self, store, row, pending):
        self.store, self.row, self.pending = store, row, pending

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.index_add_:
            require(len(args) == 4 and not kwargs and args[1] == 0,
                    'Unexpected reference combine call signature')
            source = args[3]
            require(id(source) in self.pending and self.pending[id(source)][0] is source,
                    'Actual scatter source is not the captured expert output')
            _, expert = self.pending.pop(id(source))
            expert['scatter_index'] = self.store.tensor(args[2])
            expert['scatter_source'] = self.store.tensor(source)
            expert['scatter_source_identity_verified'] = True
            result = func(*args, **kwargs)
            require(result is args[0], 'Reference in-place scatter return identity differs')
            expert['output_after_scatter'] = self.store.tensor(result)
            return result
        result = func(*args, **kwargs)
        if func is torch.nn.functional.linear and not self.row['router']:
            require('router_projection' not in self.row, 'Multiple pre-routing linear observations')
            self.row['router_projection'] = {
                'inputs': self.store.tensor(args[0]), 'weight': self.store.tensor(args[1]),
                'output': self.store.tensor(result), 'function': function_identity(func)}
        return result


def run_reference(ctx):
    import contract_moe_reference as base
    import moe_fa2_reference as fa2
    require(ctx.stage['configuration']['options'] == {'profile': 'moe_fa2'},
            'W1 dispatch observer requires the published moe_fa2 profile')
    store = Store(ctx, 'reference')
    store.sources = verify_modules({'contract_moe_reference': base, 'moe_fa2_reference': fa2}, REFERENCE_PINS)
    restore, active = Restore(), {'row': None, 'pending': {}}
    original_routes, original_install = base._State.routes, base._State.install_forward
    state = {'side': 'reference', 'restored': False}
    ctx.metadata['moe_w1_hooks'] = state

    def routes(self, name, module, inputs, logits, scores, selected, probabilities, effective):
        result = original_routes(self, name, module, inputs, logits, scores, selected, probabilities, effective)
        row = active['row']
        require(row is not None and not row['router'], 'Reference route observation lacks block context')
        row['router'] = {key: store.tensor(value) for key, value in {
            'inputs': inputs, 'weight': module.gate.weight.detach().bfloat16(),
            'logits': logits, 'selection_scores': scores, 'selected': selected,
            'selected_probabilities': probabilities, 'effective_probabilities': effective}.items()}
        row['router']['selection_domain'] = 'logits'
        row['router']['probability_rule'] = 'softmax_selected_logits'
        row['router']['parameter_name'] = name + '.gate.weight'
        row['router']['expert_ids'] = list(range(len(module.experts)))
        row['router']['expert_parameter_prefixes'] = [name + '.experts.' + str(i) for i in range(len(module.experts))]
        return result

    def install(self, module, name, strategy, function):
        if strategy == 'independent_topk_expert_dispatch':
            def sparse(instance, *args, **kwargs):
                require(active['row'] is None and not active['pending'], 'Nested reference sparse block')
                sample = ctx.suite['description']['inputs'][self.step]['samples'][self.micro - 1]
                row = store.start(self.step, int(name.split('.')[2]) + 1, sample['id'], range(args[0].shape[1]))
                row['local_expert_indices'] = list(range(len(instance.experts)))
                active['row'] = row
                try:
                    with ReferenceMode(store, row, active['pending']):
                        result = function(instance, *args, **kwargs)
                    require(not active['pending'], 'Expert output was not combined by actual scatter')
                    require('router_projection' in row, 'Actual reference router projection was not observed')
                    row['combined_output'] = store.tensor(result[0].reshape(-1, result[0].shape[-1]))
                    row['combine_boundary'] = 'actual expert-major index_add_; expert outputs already weighted'
                    store.commit(row)
                    return result
                finally:
                    active['row'] = None
                    active['pending'].clear()
            function_to_install = sparse
        elif strategy == 'joint_fc1_fp32_weighted_swiglu_before_fc2':
            expert_index = int(name.rsplit('.', 1)[1])
            def expert(instance, inputs, probabilities):
                row = active['row']
                require(row is not None, 'Reference expert has no sparse block context')
                event = {'expert_id': expert_index, 'inputs': store.tensor(inputs),
                         'probabilities': store.tensor(probabilities)}
                result = function(instance, inputs, probabilities)
                event['outputs'] = store.tensor(result)
                require(id(result) not in active['pending'], 'Expert output identity reused before combine')
                active['pending'][id(result)] = (result, event)
                row['experts'].append(event)
                return result
            function_to_install = expert
        else:
            function_to_install = function
        return original_install(self, module, name, strategy, function_to_install)

    restore.replace(base._State, 'routes', routes)
    restore.replace(base._State, 'install_forward', install)
    try:
        fa2.run_reference(ctx)
    finally:
        restore.close()
        state['restored'] = base._State.routes is original_routes and base._State.install_forward is original_install
    store.finish()


def native_scope(layer):
    config, dispatcher = layer.config, layer.token_dispatcher
    require(dispatcher.tp_size == dispatcher.ep_size == 1 and dispatcher.num_local_experts == 4,
            'W1 observer requires all four experts locally')
    require(list(dispatcher.local_expert_indices) == [0, 1, 2, 3], 'Unexpected local expert domain')
    forbidden = ('moe_permute_fusion', 'moe_router_fusion', 'moe_pad_expert_input_to_capacity',
                 'moe_apply_probs_on_input', 'moe_router_pre_softmax', 'moe_router_enable_expert_bias',
                 'moe_enable_routing_replay', 'moe_router_padding_for_quantization',
                 'fp8', 'fp4', 'use_te_activation_func', 'fine_grained_activation_offloading')
    require(not any(getattr(config, key, False) for key in forbidden),
            'Unsupported actual native mode for W1 dispatch observation')
    require(getattr(config, 'recompute_granularity', None) is None
            and getattr(config, 'moe_expert_capacity_factor', None) is None
            and getattr(config, 'moe_latent_size', None) is None
            and getattr(config, 'moe_shared_expert_intermediate_size', None) is None
            and getattr(config, 'moe_router_num_groups', None) is None
            and getattr(config, 'moe_router_group_topk', None) is None
            and getattr(config, 'moe_router_topk_scaling_factor', None) is None,
            'Unsupported actual native routing/recompute/domain configuration')
    require(config.moe_router_topk == 2 and config.num_moe_experts == 4
            and config.moe_router_score_function == 'softmax' and config.moe_router_dtype == 'fp32'
            and not config.moe_aux_loss_coeff and not config.moe_z_loss_coeff
            and layer.router.bias is None and layer.router.expert_bias is None
            and layer.router.router_replay is None and layer.router.routing_type != 'sinkhorn',
            'Unsupported actual router contract')
    return {name: getattr(config, name, None) for name in forbidden} | {
        'topk': config.moe_router_topk, 'score_function': config.moe_router_score_function,
        'router_dtype': config.moe_router_dtype, 'routing_type': layer.router.routing_type}


class NativeHooks:
    """Boundaries retain actual tensors and coordinate arguments, without a route oracle."""
    def __init__(self, store, dispatcher_module):
        self.store, self.dispatcher_module = store, dispatcher_module
        self.restore, self.active, self.phase = Restore(), None, None

    @contextmanager
    def phase_scope(self, name):
        prior = self.phase
        self.phase = name
        try:
            yield
        finally:
            self.phase = prior

    def install_helpers(self):
        for kind in ('permute', 'sort_chunks_by_idxs', 'unpermute'):
            original = getattr(self.dispatcher_module, kind)
            signature = inspect.signature(original)
            def call(*args, original=original, signature=signature, kind=kind, **kwargs):
                if self.active is None:
                    return original(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                event = {'kind': kind, 'phase': self.phase, 'arguments': {}}
                for name, value in bound.arguments.items():
                    if isinstance(value, torch.Tensor):
                        event['arguments'][name] = self.store.tensor(value)
                    elif isinstance(value, torch.Size):
                        event['arguments'][name] = list(value)
                    else:
                        require(value is None or type(value) in (bool, int, float, str),
                                'Unsupported dispatcher helper argument type')
                        event['arguments'][name] = value
                result = original(*args, **kwargs)
                values = result if isinstance(result, tuple) else (result,)
                event['returns'] = [self.store.tensor(value) if value is not None else None for value in values]
                self.active['helpers'].append(event)
                return result
            self.restore.replace(self.dispatcher_module, kind, call)

    def install_layer(self, layer, *, step, sample_id, positions, module_name=''):
        store, dispatcher, router, experts = self.store, layer.token_dispatcher, layer.router, layer.experts
        modes = native_scope(layer)
        layer_id = int(router.layer_number)
        gating, routing = router.gating, router.routing

        def gate(inputs, *args, **kwargs):
            require(self.active is None, 'Native sparse block was not fully combined')
            row = store.start(step, layer_id, sample_id(), positions)
            row['native_modes'] = modes
            row['local_expert_indices'] = [int(i) for i in dispatcher.local_expert_indices]
            self.active = row
            row['router'] = {'inputs': store.tensor(inputs.reshape(-1, inputs.shape[-1])),
                             'weight': store.tensor(router.weight),
                             'gating_function': function_identity(gating),
                             'parameter_name': module_name + '.router.weight',
                             'expert_ids': [int(i) for i in dispatcher.local_expert_indices],
                             'expert_id_provenance': 'actual local_expert_indices; pinned W1 router row and grouped-parameter mapping'}
            result = gating(inputs, *args, **kwargs)
            row['router']['gating_output'] = store.tensor(result)
            return result

        def route(logits, *args, **kwargs):
            require(self.active is not None, 'Native routing lacks actual gating context')
            result = routing(logits, *args, **kwargs)
            self.active['router'].update({
                'logits': store.tensor(logits.reshape(-1, logits.shape[-1])),
                'probabilities': store.tensor(result[0]), 'routing_map': store.tensor(result[1]),
                'selection_domain': 'logits', 'probability_rule': 'softmax_selected_logits',
                'native_topk_slot_order_observed': False})
            return result

        self.restore.replace(router, 'gating', gate)
        self.restore.replace(router, 'routing', route)
        for name in ('dispatch_preprocess', 'token_dispatch', 'dispatch_postprocess',
                     'combine_preprocess', 'token_combine', 'combine_postprocess'):
            original = getattr(dispatcher, name)
            def stage(*args, name=name, original=original, **kwargs):
                require(self.active is not None and self.active['layer'] == layer_id,
                        'Native dispatcher lacks its actual layer context')
                event = {'name': name, 'inputs': [store.tensor(v) for v in args if isinstance(v, torch.Tensor)]}
                with self.phase_scope(name):
                    result = original(*args, **kwargs)
                values = result if isinstance(result, tuple) else (result,)
                event['outputs'] = [store.tensor(v) if isinstance(v, torch.Tensor) else None for v in values]
                self.active['stages'].append(event)
                if name == 'combine_postprocess':
                    self.active['combined_output'] = store.tensor(result.reshape(-1, result.shape[-1]))
                    self.active['combine_boundary'] = 'actual unpermute input/output; expert outputs already weighted'
                    store.commit(self.active)
                    self.active = None
                return result
            self.restore.replace(dispatcher, name, stage)
        original_experts = experts.forward
        def expert_call(inputs, counts, probabilities):
            require(self.active is not None and self.active['layer'] == layer_id,
                    'Native experts lack actual layer context')
            event = {'inputs': store.tensor(inputs), 'counts': store.tensor(counts),
                     'probabilities': store.tensor(probabilities),
                     'local_expert_indices': [int(i) for i in dispatcher.local_expert_indices]}
            result = original_experts(inputs, counts, probabilities)
            require(isinstance(result, tuple) and len(result) == 2 and result[1] is None,
                    'Unsupported native expert output/bias boundary')
            event['outputs'] = store.tensor(result[0])
            self.active['experts'].append(event)
            return result
        self.restore.replace(experts, 'forward', expert_call)

    def close(self):
        self.restore.close()
        self.active = None


def run_slime(ctx):
    from slime_parity.adapters import slime as adapter
    store = Store(ctx, 'native')
    original_capture = adapter.ForwardCapture
    state = {'side': 'native', 'restored': False, 'steps_installed': 0, 'steps_restored': 0}
    ctx.metadata['moe_w1_hooks'] = state

    class Capture(original_capture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observer = None

        def install(self, modules, padded_length, topology, layout='bshd', pad_multiplier=1):
            require(topology.world_size == topology.tp == topology.cp == topology.ep == topology.pp
                    == topology.etp == topology.vpp == 1 and not topology.sp and pad_multiplier == 1,
                    'W1 dispatcher observation requires a single unpadded GPU')
            from megatron.core.transformer.moe import token_dispatcher, moe_utils, router, experts, moe_layer
            imports = dict(token_dispatcher=token_dispatcher, moe_utils=moe_utils, router=router,
                           experts=experts, moe_layer=moe_layer)
            store.sources = verify_modules(imports, NATIVE_PINS)
            store.sources['router_gating_linear'] = function_identity(router.router_gating_linear)
            store.sources['router_gating_forward'] = function_identity(moe_utils.RouterGatingLinearFunction.forward)
            store.sources['HAVE_TE'] = bool(getattr(moe_utils, 'HAVE_TE', False))
            te_gemm = getattr(moe_utils, 'te_general_gemm', None)
            store.sources['te_general_gemm_available'] = te_gemm is not None
            store.sources['te_general_gemm'] = function_identity(te_gemm) if te_gemm is not None else None
            store.sources['gating_branch_from_pinned_function'] = 'te_general_gemm' if te_gemm is not None else 'torch.mm'
            super().install(modules, padded_length, topology, layout, pad_multiplier)
            self.observer = NativeHooks(store, token_dispatcher)
            state['steps_installed'] += 1
            self.observer.install_helpers()
            step = len(ctx.steps)
            length = len(ctx.suite['description']['inputs'][step]['samples'][0]['tokens'])
            positions = adapter.native_token_positions(length, topology, 0, layout, pad_multiplier)
            count = 0
            for chunk, model in enumerate(modules):
                for module_name, layer in model.named_modules():
                    if not isinstance(layer, moe_layer.MoELayer):
                        continue
                    require(type(layer.token_dispatcher) is token_dispatcher.MoEAlltoAllTokenDispatcher
                            and isinstance(layer.experts, experts.TEGroupedMLP),
                            'Unexpected actual native dispatcher/expert implementation')
                    self.observer.install_layer(layer, step=step,
                        sample_id=lambda chunk=chunk: self.current[chunk], positions=positions, module_name=module_name)
                    count += 1
            require(count == ctx.suite['description']['constraints']['num_layers'],
                    'Native MoE layer installation coverage differs')

        def close(self):
            if self.observer is not None:
                self.observer.close()
                state['steps_restored'] += 1
            super().close()

    adapter.ForwardCapture = Capture
    try:
        adapter.run_slime(ctx)
    finally:
        adapter.ForwardCapture = original_capture
        state['restored'] = adapter.ForwardCapture is original_capture and state['steps_installed'] == state['steps_restored']
    store.finish()
