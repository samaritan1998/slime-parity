"""Independent packed-projection and one-tile FA2 arithmetic diagnostics.

This keeps HF parameters and the public canonical map. Joint projection weights
are assembled with explicit PyTorch views/cats; no native kernel supplies an
expected value. These controls do not establish acceptance tolerances.
"""
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from training_parity.common import digest, require
import contract_reference as base
import bf16_ops as ops
import attention_contract_probe as attention

BASE_SHA256='f86e608f2d5dbffb354865be72ece1e28ba480e4f0060303f1071cd1c2a7bb83'
ATTENTION_SHA256='ee1107a2f5100d23becefb8f00a183121dadf434d9579c03e98dbe4b43f88c07'

def joint_qkv(inputs, query_weight, key_weight, value_weight, head_dim):
    require(query_weight.ndim==key_weight.ndim==value_weight.ndim==2
            and key_weight.shape==value_weight.shape
            and query_weight.shape[1]==key_weight.shape[1]==inputs.shape[-1], 'Joint QKV shape differs')
    require(type(head_dim) is int and head_dim>0 and key_weight.shape[0]%head_dim==0,
            'Invalid QKV head width')
    groups=key_weight.shape[0]//head_dim
    require(query_weight.shape[0]%(groups*head_dim)==0, 'Query groups do not divide heads')
    query_width=query_weight.shape[0]//groups
    weights=torch.cat((query_weight.reshape(groups,query_width,-1),
                       key_weight.reshape(groups,head_dim,-1),
                       value_weight.reshape(groups,head_dim,-1)),dim=1)
    output=ops.linear(inputs,weights.reshape(-1,inputs.shape[-1]))
    grouped=output.reshape(*inputs.shape[:-1],groups,query_width+2*head_dim)
    q,k,v=grouped.split((query_width,head_dim,head_dim),dim=-1)
    return (q.reshape(*inputs.shape[:-1],-1),k.reshape(*inputs.shape[:-1],-1),
            v.reshape(*inputs.shape[:-1],-1))

def joint_gate_up(inputs, gate_weight, up_weight):
    require(gate_weight.shape==up_weight.shape, 'Gate/up shapes differ')
    return ops.linear(inputs,torch.cat((gate_weight,up_weight),0)).chunk(2,-1)

class _FA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,query,key,value):
        require(query.ndim==key.ndim==value.ndim==4 and query.shape[0]==key.shape[0]==value.shape[0]==1,
                'One-sequence FA2 diagnostic only')
        q,k,v=[x[0].permute(1,0,2).contiguous() for x in (query,key,value)]
        result=attention.evaluate_attention(q,k,v,torch.zeros_like(q),variant='fa2_contract')
        ctx.save_for_backward(q,k,v)
        return result['forward'].permute(1,0,2)[None]

    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        q,k,v=ctx.saved_tensors
        result=attention.evaluate_attention(q,k,v,upstream[0].permute(1,0,2).contiguous(),variant='fa2_contract')
        return tuple(result[name].permute(1,0,2)[None] for name in ('q_gradient','k_gradient','v_gradient'))

def fa2_attention(query,key,value,*,causal=True,scale=None):
    require(causal and (scale is None or scale==query.shape[-1]**-.5), 'FA2 scale/mask differs')
    return _FA2.apply(query,key,value)

class _ProfileContext:
    def __init__(self,original,profile):
        self.original=original
        self.stage={**original.stage,'configuration':{**original.stage['configuration'],'options':{'profile':profile}}}
    def __getattr__(self,name):
        return getattr(self.original,name)

def run_reference(ctx):
    import transformers.models.qwen3.modeling_qwen3 as qwen
    options=ctx.stage['configuration']['options']
    require(set(options)=={'profile'} and options['profile'] in ('packed_eager','packed_math','packed_fa2'),
            'Unknown packed arithmetic diagnostic profile')
    require(digest(Path(base.__file__))==BASE_SHA256 and digest(Path(attention.__file__))==ATTENTION_SHA256,
            'Packed arithmetic dependencies changed')
    profile=options['profile']
    metadata={'profile':profile,'purpose':'packed BF16 arithmetic diagnostic; not acceptance qualification',
              'joint_qkv':'group-major Q,K,V joint GEMM with one BF16 input-gradient writeback',
              'joint_mlp':'gate/up joint GEMM with one BF16 input-gradient writeback',
              'attention':'HF eager' if profile=='packed_eager' else ('FP32 mathematical' if profile=='packed_math' else 'independent one-tile FA2 source boundaries'),
              'base_sha256':BASE_SHA256,'attention_helper_sha256':ATTENTION_SHA256,'restored':False}
    ctx.metadata['packed_arithmetic']=metadata
    original_install=base._install
    original_attention=ops.attention_fp32

    def install(model,state,qwen_module):
        original_install(model,state,qwen_module)
        # The base installer has already selected the numerical operations.
        # Record the outer profile in every step, without changing those ops.
        state.metadata['base_profile']=state.profile
        state.profile=profile
        state.metadata['profile']=state.metadata['requested_profile']=profile
        state.metadata['attention']=metadata['attention']
        for i,layer in enumerate(model.model.layers):
            prefix=f'model.layers.{i}'
            attention_name=prefix+'.self_attn'
            mlp_name=prefix+'.mlp'
            def attn_forward(module,hidden_states,position_embeddings,attention_mask,
                             past_key_values=None,cache_position=None,_name=attention_name,**kwargs):
                require(past_key_values is None and not kwargs.get('use_cache',False),'Cache unsupported')
                for child in ('q_proj','k_proj','v_proj'):
                    state.counts[_name+'.'+child]+=1
                q,k,v=joint_qkv(hidden_states,module.q_proj.weight,module.k_proj.weight,module.v_proj.weight,module.head_dim)
                shape=(*hidden_states.shape[:-1],-1,module.head_dim)
                q=module.q_norm(q.reshape(shape)).transpose(1,2)
                k=module.k_norm(k.reshape(shape)).transpose(1,2)
                v=v.reshape(shape).transpose(1,2)
                q,k=qwen_module.apply_rotary_pos_emb(q,k,*position_embeddings)
                output,weights=qwen_module.eager_attention_forward(module,q,k,v,attention_mask,
                    scaling=module.scaling,dropout=0.0)
                return module.o_proj(output.reshape(*hidden_states.shape[:-1],-1).contiguous()),weights
            def mlp_forward(module,hidden_states,_name=mlp_name):
                for child in ('gate_proj','up_proj'):
                    state.counts[_name+'.'+child]+=1
                gate,up=joint_gate_up(hidden_states,module.gate_proj.weight,module.up_proj.weight)
                return module.down_proj(ops.swiglu(gate,up))
            state.install_forward(layer.self_attn,attention_name,'joint_group_major_qkv',attn_forward)
            state.install_forward(layer.mlp,mlp_name,'joint_gate_up_fp32_swiglu',mlp_forward)
            for child in ('q_proj','k_proj','v_proj'):
                state.strategies[attention_name+'.'+child]='slice_of_joint_qkv_fp32_wgrad'
            for child in ('gate_proj','up_proj'):
                state.strategies[mlp_name+'.'+child]='slice_of_joint_gate_up_fp32_wgrad'
        if profile=='packed_fa2':
            for name in list(state.strategies):
                if name.endswith('.math_attention'):
                    state.strategies[name]='independent_one_tile_fa2_source_boundaries'
            state.metadata['limits'].append('FA2 source boundaries cover D32 and length<=128; not CUDA instruction emulation or CP backend qualification.')
            state.metadata['limits']=[item for item in state.metadata['limits']
                if item!='Math attention is not a FlashAttention implementation or backend contract.']
            state.metadata['attention_counter_label']='Inherited *.math_attention counters count the independent one-tile FA2 callable.'

    base._install=install
    if profile=='packed_fa2':
        ops.attention_fp32=fa2_attention
    try:
        base.run_reference(_ProfileContext(ctx,'fused_norm_rope_swiglu' if profile=='packed_eager' else 'math_attention'))
    finally:
        base._install=original_install
        ops.attention_fp32=original_attention
        if 'reference_precision' in ctx.metadata:
            ctx.metadata['reference_precision']['attention']=metadata['attention']
        metadata['restored']=base._install is original_install and ops.attention_fp32 is original_attention
