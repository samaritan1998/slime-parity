import pytest
import sys
from types import ModuleType, SimpleNamespace

from slime_parity.adapters.slime import replace_arguments, validate_native_precision, validate_native_sft_objective
from training_parity.common import EvidenceError


def test_replaces_duplicate_topology_flags_without_shell_parsing():
    args=['--model-name','a model','--tensor-model-parallel-size','8','--tensor-model-parallel-size=4','--sequence-parallel','--lr','0.1']
    result=replace_arguments(args,{'--tensor-model-parallel-size':2,'--lr':0.01},{'--sequence-parallel':False})
    assert result==['--model-name','a model','--tensor-model-parallel-size','2','--lr','0.01']


def test_clears_vpp_and_preserves_unrelated_arguments():
    result=replace_arguments(['--num-layers-per-virtual-pipeline-stage','2','--custom','a;b'],
                             {'--num-layers-per-virtual-pipeline-stage':None},{'--sequence-parallel':True})
    assert result==['--custom','a;b','--sequence-parallel']


def test_rejects_malformed_argument():
    with pytest.raises(EvidenceError):replace_arguments(['--lr','--other'],{'--lr':1},{})


@pytest.mark.parametrize("requested,bf16,actual", [("float32", True, "bfloat16"),
                                                  ("bfloat16", False, "float32"),
                                                  ("float32", False, "bfloat16")])
def test_effective_precision_drift_is_rejected(requested, bf16, actual):
    import torch
    args = SimpleNamespace(fp16=False, bf16=bf16, params_dtype=getattr(torch, actual))
    with pytest.raises(EvidenceError, match="effective precision"):
        validate_native_precision(args, {"dtype": requested})


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_loaded_native_parameters_match_declared_precision(dtype, monkeypatch):
    import torch
    expected = getattr(torch, dtype)
    args = SimpleNamespace(fp16=False, bf16=dtype=="bfloat16", params_dtype=expected)
    utils = ModuleType("megatron.core.utils")
    utils.get_model_config = lambda model: SimpleNamespace(params_dtype=expected)
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    model = torch.nn.Linear(2, 3, bias=False, dtype=expected)
    result = validate_native_precision(args, {"dtype": dtype}, [model])
    assert result["parameter_elements"] == {str(expected): 6}
    model.weight.data = model.weight.data.to(torch.float64)
    with pytest.raises(EvidenceError, match="parameter precision"):
        validate_native_precision(args, {"dtype": dtype}, [model])


@pytest.mark.parametrize("temperature", [.5, 0, -1, float("nan")])
def test_native_sft_rejects_scaled_logits(temperature):
    args = SimpleNamespace(loss_type="sft_loss", calculate_per_token_loss=True, rollout_temperature=temperature)
    with pytest.raises(EvidenceError, match="rollout_temperature=1"):
        validate_native_sft_objective(args)


def test_native_sft_accepts_matching_objective():
    validate_native_sft_objective(SimpleNamespace(loss_type="sft_loss", calculate_per_token_loss=True, rollout_temperature=1.0))
