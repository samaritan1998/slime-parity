from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from slime_parity.adapters.hf import normalized_inputs, validate_zero_dropout, HFCausalLMAdapter
from slime_parity.adapters.mlp import example_spec
from slime_parity.suite import prepare, load_suite, check_capture
from training_parity.common import EvidenceError, write_new, read
from training_parity.runner import run_stage
from training_parity.storage import Stage, aligned_chunks


def text_inputs():
    return [{"samples": [{"id": i, "tokens": [1, 2+i, 3+step, 4, 5, 6],
                           "loss_mask": [0, 0, 1, 1, 1, 1]} for i in range(2)]} for step in range(3)]


@pytest.mark.parametrize("field", ["attn_pdrop", "resid_pdrop", "embd_pdrop"])
def test_gpt2_dropout_is_rejected_during_model_description(tmp_path, field):
    pytest.importorskip("transformers")
    from transformers import GPT2Config
    config = GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2,
                        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, bos_token_id=1, eos_token_id=2)
    setattr(config, field, .5)
    config.save_pretrained(tmp_path / "checkpoint")
    adapter = HFCausalLMAdapter({"checkpoint": str(tmp_path/"checkpoint"), "parameter_map": str(tmp_path/"unused.json")})
    with pytest.raises(EvidenceError, match=field):
        adapter.describe({"id": "text", "inputs": text_inputs()}, {"steps":3, "global_batch_size":2})


def test_actual_module_dropout_is_checked_even_when_config_is_zero():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2,
                                      attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, bos_token_id=1, eos_token_id=2))
    validate_zero_dropout(model)
    module = next(m for m in model.modules() if isinstance(m, torch.nn.Dropout))
    module.p = .5
    with pytest.raises(EvidenceError, match="zero dropout"):
        validate_zero_dropout(model)


@pytest.mark.parametrize("fault", ["mask", "tokens", "order", "steps", "ragged"])
def test_frozen_text_inputs_reject_semantic_mismatches(fault):
    inputs = text_inputs()
    if fault == "mask": inputs[0]["samples"][0]["loss_mask"][0] = 1
    if fault == "tokens": inputs[0]["samples"][0]["tokens"][0] = 16
    if fault == "order": inputs[0]["samples"].reverse()
    if fault == "steps": inputs.pop()
    if fault == "ragged":
        inputs[0]["samples"][0]["tokens"].pop()
        inputs[0]["samples"][0]["loss_mask"].pop()
    with pytest.raises(EvidenceError):
        normalized_inputs({"id": "text", "inputs": inputs}, {"steps": 3, "global_batch_size": 2}, 16)


def test_local_hf_reference_runs_three_steps_with_complete_evidence(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(71)
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2,
                         attention_dropout=0, tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    model.save_pretrained(tmp_path / "checkpoint", safe_serialization=True)
    names = {name: name for name,p in model.named_parameters() if p.requires_grad}
    write_new(tmp_path / "mapping.json", names)
    spec = example_spec()
    spec["model"] = {"adapter": "slime_parity.adapters.hf:HFCausalLMAdapter",
                     "options": {"checkpoint": "checkpoint", "parameter_map": "mapping.json"}}
    spec["reference"] = {"runner": "slime_parity.adapters.hf:run_reference", "kind": "independent", "device": "cpu"}
    # The candidate is deliberately not executed by this independent-reference
    # test. It supplies no evidence of Slime or model-parallel compatibility.
    spec["training"].update(dtype="float32", global_batch_size=2)
    spec["training"]["optimizer"]["weight_decay"] = 0
    spec["matrix"]["world_size"] = [1]
    spec["cases"] = [{"id": "text", "inputs": text_inputs()}]
    write_new(tmp_path / "spec.json", spec)
    path = prepare(tmp_path / "spec.json", tmp_path / "suite")
    suite = load_suite(path)
    assert suite["spec"]["model"]["options"]["checkpoint"] == str(tmp_path / "checkpoint")
    plan = path.parent / suite["cases"][0]["plan"]
    run_stage(plan, "reference", timeout=90)
    stage = Stage(plan, "reference")
    check_capture(stage)
    assert len(stage.ranks[0]["metadata"]["steps"]) == 3
    # Check a forward value independently against the original loaded weights.
    with torch.no_grad():
        tokens = torch.tensor([s["tokens"] for s in text_inputs()[0]["samples"]])
        model.config._attn_implementation = "eager"
        logits = model(tokens, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, 1:5].flatten(0,1), tokens[:, 2:6].flatten())
    actual = np.concatenate([left for left, _ in aligned_chunks(stage, stage, (0, "observation/loss"))])
    np.testing.assert_allclose(actual, [loss.item()], rtol=1e-6, atol=1e-7)
    # The model-onboarding generator must retain every parallel axis and leave
    # numerical acceptance unqualified until a real policy is supplied.
    write_new(tmp_path / "argv.json", ["--hf-checkpoint", str(tmp_path / "checkpoint")])
    generator = Path(__file__).resolve().parents[1] / "examples/make_slime_suite.py"
    subprocess.run([sys.executable, str(generator), "--checkpoint", str(tmp_path/"checkpoint"),
                    "--parameter-map", str(tmp_path/"mapping.json"), "--slime-source", str(tmp_path),
                    "--megatron-source", str(tmp_path), "--slime-args", str(tmp_path/"argv.json"),
                    "--output", str(tmp_path/"generated.json")], check=True, capture_output=True)
    generated = read(tmp_path/"generated.json")
    from slime_parity.suite import normalize
    from slime_parity.topology import expand_matrix
    generated = normalize(generated)
    assert generated["policy"] == [] and generated["training"]["steps"] == 3
    assert len(generated["cases"]) == 2
    for case in generated["cases"]:
        normalized_inputs(case, generated["training"], 16)
    assert len(expand_matrix(generated["matrix"], {"num_layers":2, "num_experts":0}, 16)) == 128
