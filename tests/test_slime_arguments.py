import pytest
import os
import sys
from types import ModuleType, SimpleNamespace

from slime_parity.adapters.slime import (replace_arguments, validate_native_precision,
                                        validate_native_sft_objective, validate_native_compute_modes)
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


@pytest.mark.parametrize("flag", ["OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "ENABLE_ROUTING_REPLAY"])
def test_native_rejects_hidden_compute_modes_even_with_bf16_parameters(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(EvidenceError, match="dedicated adapter"):
        validate_native_compute_modes(SimpleNamespace(bf16=True, fp8=None, fp4=None))
    assert os.environ[flag] == "1"  # Never silently disable the requested mode.


@pytest.mark.parametrize("field,value", [("fp8", "hybrid"), ("fp4", "nvfp4"),
    ("fp8_param", True), ("fp8_param_gather", True), ("fp4_param", True), ("use_kitchen", True),
    ("kitchen_config_file", "recipe.yaml"), ("kitchen_recipe_number", 0),
    ("activation_func_fp8_input_store", True), ("moe_router_force_load_balancing", True),
    ("moe_router_force_biased", 0), ("use_routing_replay", True), ("use_rollout_routing_replay", True)])
def test_native_rejects_declared_quantized_or_forced_compute(monkeypatch, field, value):
    monkeypatch.delenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", raising=False)
    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    with pytest.raises(EvidenceError, match="dedicated adapter"):
        validate_native_compute_modes(SimpleNamespace(**{field: value}))


def test_inactive_quantization_and_replay_settings_do_not_change_compute(monkeypatch):
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("OPEN_TRAINING_INT4_GROUP_SIZE", "128")
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "0")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    assert validate_native_compute_modes(SimpleNamespace()) == {
        "fp8": None, "fp4": None, "fp8_param": None, "fp8_param_gather": None,
        "fp4_param": None, "use_kitchen": None, "requested_kitchen_recipe": False,
        "activation_func_fp8_input_store": None,
        "moe_router_force_load_balancing": None, "forced_router_bias": None,
        "fake_int4": False, "routing_replay": False, "requested_routing_replay": False}


def test_native_checks_actual_model_quantization_overrides(monkeypatch):
    import torch
    monkeypatch.delenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", raising=False)
    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    config = SimpleNamespace(activation_func_fp8_input_store=False)
    utils = ModuleType("megatron.core.utils")
    utils.get_model_config = lambda model: config
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))
    recipe = SimpleNamespace(override_nonquantized_autocast=False,
                             fp8_quantization_recipe=object(), fp4_quantization_recipe=None)
    model[0].te_quant_params = SimpleNamespace(training_recipe=recipe)
    result = validate_native_compute_modes(SimpleNamespace(fp8_recipe="delayed"), [model])
    assert result["models"][0]["module_quantization_overrides"][0]["module"] == "0"
    recipe.override_nonquantized_autocast = True
    with pytest.raises(EvidenceError, match="quantized training"):
        validate_native_compute_modes(SimpleNamespace(), [model])
    recipe.override_nonquantized_autocast = False
    config.activation_func_fp8_input_store = True
    with pytest.raises(EvidenceError, match="quantized activations"):
        validate_native_compute_modes(SimpleNamespace(), [model])
    config.activation_func_fp8_input_store = False
    config.fp8_param = True
    with pytest.raises(EvidenceError, match="dedicated adapter"):
        validate_native_compute_modes(SimpleNamespace(), [model])
    config.fp8_param = False
    config.use_kitchen = True
    with pytest.raises(EvidenceError, match="dedicated adapter"):
        validate_native_compute_modes(SimpleNamespace(), [model])
