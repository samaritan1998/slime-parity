"""MoE diagnostic using independent one-tile FA2 source boundaries.

The frozen MoE control still supplies autonomous routing, packed projections,
normalization, RoPE, weighted experts and the public HF optimizer/objective.
Only its mathematical-attention callable is replaced. No native attention
kernel or captured native tensor supplies an expected value.
"""
from pathlib import Path

from training_parity.common import digest, require
import attention_contract_probe as attention
import bf16_ops as ops
import contract_reference as dense
import contract_moe_reference as base
import dense_packed_reference as packed


PROFILE = 'moe_fa2'
ATTENTION_LABEL = 'independent one-tile FA2 source boundaries'
DEPENDENCIES = {
    'contract_moe_reference.py': '21a7acbaf343e371763dfbe8806c1ad58c591b8162a55d1544c3c55c7129145e',
    'dense_packed_reference.py': 'c01e8dc5144aa9fd3c8a372462505f3226c6e523736be6d22660185bb1c5e703',
    'attention_contract_probe.py': 'ee1107a2f5100d23becefb8f00a183121dadf434d9579c03e98dbe4b43f88c07',
    'contract_reference.py': 'f86e608f2d5dbffb354865be72ece1e28ba480e4f0060303f1071cd1c2a7bb83',
    'bf16_ops.py': 'd4e0eadb72cdd35ea53090e10b684784b375bf0d0882d0f4200d0676ea18b42f',
    'slime_parity/adapters/hf.py': '48b96a757a280a96e256c19485ecda85a650dc6afaf5fe8928b37708df5c8cf8',
}


def run_reference(ctx):
    from slime_parity.adapters import hf

    options = ctx.stage['configuration']['options']
    require(isinstance(options, dict) and options == {'profile': PROFILE}, 'Unknown MoE FA2 overlay option')
    sources = {'contract_moe_reference.py': base, 'dense_packed_reference.py': packed,
               'attention_contract_probe.py': attention, 'contract_reference.py': dense,
               'bf16_ops.py': ops, 'slime_parity/adapters/hf.py': hf}
    require(all(digest(Path(module.__file__)) == DEPENDENCIES[name] for name, module in sources.items()),
            'Frozen MoE FA2 overlay dependency changed')
    metadata = {
        'profile': PROFILE, 'purpose': 'independent arithmetic diagnostic; not acceptance qualification',
        'attention': ATTENTION_LABEL, 'base_control': 'MoE math_attention with attention overridden',
        'source_sha256': digest(Path(__file__)), 'dependencies': dict(DEPENDENCIES),
        'scope': {'head_dim': 32, 'minimum_length': 1, 'maximum_length': 128,
                  'micro_batch_size': 1, 'causal': True, 'dropout': 0, 'cache': False},
        'limits': ['Single-tile source-boundary formula, not CUDA instruction or reduction-order emulation.',
                   'Original fixed policy is unchanged; an improved numerical result is not qualification.',
                   'No inference about multi-tile sequences or CP/EP/ETP/TP/PP communication follows from this reference.'],
        'attention_forward_calls': 0, 'restored': False,
    }
    ctx.metadata['moe_fa2_overlay'] = metadata
    original_stage = ctx.stage
    stage_was_instance_attribute = 'stage' in vars(ctx)
    original_install, original_attention = base._install, ops.attention_fp32

    def install(model, state, qwen):
        require(all(layer.self_attn.head_dim == 32 for layer in model.model.layers),
                'MoE FA2 overlay requires D32')
        require(all(1 <= len(sample['tokens']) <= 128
                    for batch in ctx.suite['description']['inputs'] for sample in batch['samples']),
                'MoE FA2 overlay requires 1 <= sequence length <= 128')
        original_install(model, state, qwen)
        state.profile = PROFILE
        state.metadata['base_profile'] = 'math_attention'
        state.metadata['profile'] = state.metadata['requested_profile'] = PROFILE
        state.metadata['attention'] = ATTENTION_LABEL
        state.metadata['attention_helper_sha256'] = DEPENDENCIES['attention_contract_probe.py']
        for name in state.strategies:
            if name.endswith('.math_attention'):
                state.strategies[name] = 'independent_one_tile_fa2_source_boundaries'
        state.metadata['attention_counter_label'] = 'Inherited *.math_attention counters count the FA2 source-boundary callable in this overlay.'
        state.metadata['limits'] = [
            item for item in state.metadata['limits']
            if item not in ('Math attention is not FA2/FA3; other profiles retain HF eager attention.',
                            'Math profile also replaces embedding with the explicit chunk32 backward formula.')
        ] + ['This overlay retains the explicit chunk32 embedding backward formula.',
             'Attention is independent one-tile FA2 source boundaries, limited to D32 and length <= 128; not backend qualification.']

    def fa2(*args, **kwargs):
        result = packed.fa2_attention(*args, **kwargs)
        metadata['attention_forward_calls'] += 1
        return result

    # Use the caller context for sample/step attributes so external layer probes
    # observe the true microbatch. Replace the stage object only temporarily;
    # the frozen stage dictionary and nested options are never edited.
    ctx.stage = {**original_stage, 'configuration': {**original_stage['configuration'],
                                                   'options': {'profile': 'math_attention'}}}
    base._install, ops.attention_fp32 = install, fa2
    try:
        base.run_reference(ctx)
        expected = len(ctx.suite['description']['inputs']) * ctx.suite['training']['global_batch_size']
        expected *= len(ctx.metadata['arithmetic_control']['steps'][0]['routing'])
        require(metadata['attention_forward_calls'] == expected, 'MoE FA2 attention coverage differs')
    finally:
        base._install, ops.attention_fp32 = original_install, original_attention
        if stage_was_instance_attribute:
            ctx.stage = original_stage
        else:
            del ctx.stage
        precision = ctx.metadata.get('reference_precision')
        if precision is not None:
            precision['attention'] = ATTENTION_LABEL
        metadata['restored'] = (base._install is original_install and ops.attention_fp32 is original_attention
                                and ctx.stage is original_stage
                                and ('stage' in vars(ctx)) == stage_was_instance_attribute)
