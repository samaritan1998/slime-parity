from copy import deepcopy
import pytest
import torch
from transformers import AutoModelForCausalLM,Qwen3Config,Qwen3ForCausalLM
import transformers.models.qwen3.modeling_qwen3 as qwen
from test_contract_reference import Context,tiny_model,loader
import contract_reference as base
import bf16_ops as ops
import dense_packed_reference as packed

def test_joint_qkv_head_mapping_and_single_writeback_vjp():
    x=(torch.arange(10).reshape(2,5)%7-3).bfloat16().requires_grad_()
    weights=[torch.nn.Parameter(((torch.arange(n*5).reshape(n,5)+i)%11-5).float()/4)
             for i,n in enumerate((8,4,4))]
    outputs=packed.joint_qkv(x,*weights,2)
    upstream=[((torch.arange(y.numel()).reshape(y.shape)+i)%7-3).bfloat16()
              for i,y in enumerate(outputs)]
    for output,w in zip(outputs,weights):
        expected=(x.detach().double()@w.detach().bfloat16().double().T).bfloat16()
        assert torch.equal(output,expected)
    torch.autograd.backward(outputs,upstream)
    dx=sum(g.double()@w.detach().bfloat16().double() for g,w in zip(upstream,weights)).bfloat16()
    assert torch.equal(x.grad,dx)
    for w,g in zip(weights,upstream):
        assert torch.equal(w.grad,(g.double().T@x.detach().double()).float())

def test_joint_gate_up_vjp_uses_both_parameter_halves():
    x=torch.tensor([[1.,-.5,2.],[-2.,1.,.25]],dtype=torch.bfloat16,requires_grad=True)
    gate=torch.nn.Parameter(torch.arange(12).reshape(4,3).float()/8)
    up=torch.nn.Parameter(-torch.arange(12).reshape(4,3).float()/16)
    a,b=packed.joint_gate_up(x,gate,up)
    ga=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],dtype=torch.bfloat16)
    gb=-ga/2
    torch.autograd.backward((a,b),(ga,gb))
    assert torch.equal(x.grad,(ga.double()@gate.detach().double()+gb.double()@up.detach().double()).bfloat16())
    assert torch.equal(gate.grad,(ga.double().T@x.detach().double()).float())
    assert torch.equal(up.grad,(gb.double().T@x.detach().double()).float())

def model32():
    torch.manual_seed(98)
    config=Qwen3Config(vocab_size=32,hidden_size=128,intermediate_size=192,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=32,rope_theta=1_000_000,
        attention_dropout=0,tie_word_embeddings=False)
    config._attn_implementation='eager'
    return Qwen3ForCausalLM(config)

@pytest.mark.parametrize('profile',['packed_eager','packed_math','packed_fa2'])
@pytest.mark.parametrize('fail',[False,True])
def test_three_step_optimizer_coverage_and_restore(tmp_path,monkeypatch,profile,fail):
    model=model32() if profile=='packed_fa2' else tiny_model()
    ctx=Context(tmp_path,model,profile)
    initial=deepcopy(ctx.stage)
    loaded=loader(monkeypatch,model)
    original_loader=AutoModelForCausalLM.__dict__['from_pretrained']
    original_install=base._install
    original_attention=ops.attention_fp32
    original_rope=qwen.apply_rotary_pos_emb
    original_eager=qwen.eager_attention_forward
    if fail:
        def broken(*args,**kwargs): raise RuntimeError('injected joint projection failure')
        monkeypatch.setattr(packed,'joint_qkv',broken)
        with pytest.raises(RuntimeError,match='injected joint projection'):
            packed.run_reference(ctx)
    else:
        packed.run_reference(ctx)
        assert len(ctx.steps)==3
        assert len(ctx.metadata['arithmetic_control']['steps'])==3
        metadata=ctx.metadata['arithmetic_control']
        assert metadata['profile']==metadata['requested_profile']==profile
        assert all(record['profile']==profile for record in metadata['steps'])
        assert all(record[1]['details']['arithmetic_control']['profile']==profile for record in ctx.steps)
        assert metadata['attention']==ctx.metadata['reference_precision']['attention']
        assert all(torch.isfinite(v).all() for v in ctx.values.values())
        for name,p in model.named_parameters():
            assert torch.equal(ctx.values[0,'initial',name],p.detach().bfloat16().float())
            for step in (1,2):
                assert torch.equal(ctx.values[step,'initial',name],ctx.values[step-1,'parameter',name])
            for step in range(3):
                assert torch.equal(ctx.values[step,'update',name],ctx.values[step,'parameter',name]-ctx.values[step,'initial',name])
    assert ctx.stage==initial
    assert base._install is original_install and ops.attention_fp32 is original_attention
    assert AutoModelForCausalLM.__dict__['from_pretrained'] is original_loader
    assert qwen.apply_rotary_pos_emb is original_rope and qwen.eager_attention_forward is original_eager
    assert ctx.metadata['packed_arithmetic']['restored'] and ctx.metadata['arithmetic_control']['restored']
    assert all('forward' not in vars(module) for module in loaded[0].modules())
