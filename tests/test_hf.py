from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

from slime_parity.adapters.hf import normalized_inputs, validate_zero_dropout, HFCausalLMAdapter, run_reference
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


class TinyCausalLM(torch.nn.Module):
    """Small differentiable fixture; this does not claim native-model parity."""
    def __init__(self, *, tied=False, branched=False, unused=False):
        super().__init__()
        self.config = SimpleNamespace(to_dict=lambda: {})
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.branched = branched
        if tied:
            self.head.weight = self.embedding.weight
        if branched:
            self.other_head = torch.nn.Linear(4, 8, bias=False)
        if unused:
            self.unused = torch.nn.Parameter(torch.ones(2))
        # RoPE-like persistent buffers must retain their constructed precision.
        self.register_buffer("frequency", torch.tensor([0.123456789]))

    def forward(self, input_ids, use_cache=False):
        head = self.other_head if self.branched and input_ids[0, 0].item() % 2 == 0 else self.head
        return SimpleNamespace(logits=head(self.embedding(input_ids)))


def precision_inputs():
    masks = [[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1]]
    return [{"samples": [{"id": i, "tokens": [i + 1, 2, (3 + step) % 8, 4 + i, 6],
                           "loss_mask": mask} for i, mask in enumerate(masks)]} for step in range(3)]


class MemoryReference:
    def __init__(self, tmp_path, model, options=None, *, dtype="float32", clip_grad=0):
        names = {name: name for name, p in model.named_parameters() if p.requires_grad}
        write_new(tmp_path / "mapping.json", names)
        self.suite = {
            "training": {"dtype": dtype, "global_batch_size": 3, "steps": 3, "clip_grad": clip_grad,
                         "optimizer": {"name": "adamw", "lr": .0027, "betas": [.8, .95],
                                       "eps": 1e-5, "weight_decay": .02}},
            "description": {"inputs": precision_inputs(),
                            "parameters": {n: {"shape": list(p.shape)} for n, p in model.named_parameters()}}}
        self.stage = {"configuration": {"device": "cpu", "options": options or {}}}
        self.plan = {"artifacts": {"hf_checkpoint": {"path": "unused-mocked-checkpoint"},
                                   "hf_parameter_map": {"path": str(tmp_path / "mapping.json")}}}
        self.metadata, self.tensors, self.steps = {}, {}, []

    def write(self, step, role, name, value):
        key = (step, role, name)
        assert key not in self.tensors
        self.tensors[key] = value.detach().cpu().clone()

    def observe(self, step, name, value):
        self.write(step, "observation", name, value)

    def record_step(self, step, **details):
        self.steps.append({"step": step, **details})


def mock_checkpoint(monkeypatch, model):
    transformers = pytest.importorskip("transformers")
    loaded = []

    def load(*args, **kwargs):
        loaded.append(deepcopy(model))
        return loaded[-1]

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    return loaded


def legacy_reference(model, ctx):
    """Frozen pre-control full-batch behavior, optimizing model parameters directly."""
    training = ctx.suite["training"]
    parameters = dict(model.named_parameters())
    if training["dtype"] == "bfloat16":
        with torch.no_grad():
            for parameter in parameters.values():
                parameter.copy_(parameter.bfloat16().float())
    options = {key: value for key, value in training["optimizer"].items() if key != "name"}
    optimizer = torch.optim.AdamW(parameters.values(), **options, foreach=False, fused=False)
    for step, data in enumerate(ctx.suite["description"]["inputs"]):
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in parameters.items()}
        for name, value in before.items():
            ctx.write(step, "initial", name, value)
        tokens = torch.tensor([sample["tokens"] for sample in data["samples"]])
        mask = torch.tensor([sample["loss_mask"] for sample in data["samples"]], dtype=torch.float32)
        scope = (torch.autocast("cpu", dtype=torch.bfloat16)
                 if ctx.stage["configuration"]["options"].get("compute_dtype") == "bfloat16" else nullcontext())
        with scope:
            output = model(input_ids=tokens, use_cache=False).logits
            losses = torch.nn.functional.cross_entropy(output[:, :-1].float().reshape(-1, output.shape[-1]),
                                                       tokens[:, 1:].reshape(-1), reduction="none")
            loss = (losses * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        ctx.observe(step, "logits", output)
        ctx.observe(step, "loss", loss)
        loss.backward()
        for name, p in parameters.items():
            ctx.write(step, "gradient", name, p.grad)
        if training["clip_grad"]:
            torch.nn.utils.clip_grad_norm_(parameters.values(), training["clip_grad"])
        optimizer.step()
        for name, p in parameters.items():
            ctx.write(step, "parameter", name, p)
            ctx.write(step, "update", name, p.detach() - before[name])
            for field in ("exp_avg", "exp_avg_sq"):
                ctx.write(step, field, name, optimizer.state[p][field])


@pytest.mark.parametrize("dtype,compute,tied,clip", [
    ("float32", "float32", False, 0), ("float32", "float32", True, .05),
    ("bfloat16", "float32", False, 0), ("bfloat16", "bfloat16", False, .05)])
def test_default_controls_preserve_legacy_reference_bits(tmp_path, monkeypatch, dtype, compute, tied, clip):
    torch.manual_seed(127)
    model = TinyCausalLM(tied=tied)
    options = {} if compute == "float32" else {"compute_dtype": compute}
    actual = MemoryReference(tmp_path / "actual", model, options, dtype=dtype, clip_grad=clip)
    expected = MemoryReference(tmp_path / "expected", model, options, dtype=dtype, clip_grad=clip)
    loaded = mock_checkpoint(monkeypatch, model)
    run_reference(actual)
    legacy_reference(deepcopy(model), expected)
    assert actual.tensors.keys() == expected.tensors.keys()
    for key, value in actual.tensors.items():
        # Compare bytes, including signed zero, rather than approximate equality.
        assert value.dtype == expected.tensors[key].dtype
        assert torch.equal(value.reshape(-1).view(torch.uint8),
                           expected.tensors[key].reshape(-1).view(torch.uint8)), key
    if tied:
        assert loaded[0].head.weight is loaded[0].embedding.weight
        assert actual.metadata["reference_precision"]["parameter_aliases"] == {"head.weight": "embedding.weight"}


@pytest.mark.parametrize("micro_batch_size", [1, 2])
@pytest.mark.parametrize("timing", ["before_backward", "after_backward"])
def test_microbatches_use_global_masked_objective(tmp_path, monkeypatch, micro_batch_size, timing):
    torch.manual_seed(42)
    model = TinyCausalLM()
    ctx = MemoryReference(tmp_path, model, {"micro_batch_size": micro_batch_size, "normalization_timing": timing})
    mock_checkpoint(monkeypatch, model)
    run_reference(ctx)
    data = precision_inputs()[0]["samples"]
    tokens = torch.tensor([sample["tokens"] for sample in data])
    mask = torch.tensor([sample["loss_mask"] for sample in data], dtype=torch.float32)[:, 1:]
    logits = model(tokens).logits
    # Independent expression of the full global objective; unequal masks make
    # an average of per-sample/per-micro means observably wrong.
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
    nll = -logprobs.gather(-1, tokens[:, 1:, None]).squeeze(-1)
    correct = (nll * mask).sum() / 7
    wrong = ((nll * mask).sum(-1) / mask.sum(-1)).mean()
    assert abs((correct - wrong).item()) > .01
    correct.backward()
    torch.testing.assert_close(ctx.tensors[0, "observation", "logits"], logits)
    torch.testing.assert_close(ctx.tensors[0, "observation", "loss"], correct)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(ctx.tensors[0, "gradient", name], parameter.grad, atol=3e-8, rtol=2e-6)
    for step in ctx.steps:
        assert step["details"]["supervised_tokens"] == 7
        assert set(step["details"]["seen_micro_count"].values()) == {(3 + micro_batch_size - 1) // micro_batch_size}


@pytest.mark.parametrize("timing", ["before_backward", "after_backward"])
def test_bf16_model_uses_fp32_adam_masters_and_preserves_buffers(tmp_path, monkeypatch, timing):
    torch.manual_seed(84)
    model = TinyCausalLM()
    ctx = MemoryReference(tmp_path, model, {"compute_dtype": "bfloat16", "parameter_dtype": "bfloat16",
                                          "micro_batch_size": 1, "normalization_timing": timing}, dtype="bfloat16")
    loaded = mock_checkpoint(monkeypatch, model)
    run_reference(ctx)
    # Independently collect per-sample BF16 autograd results before any update.
    # Their FP32 sum must survive without a second BF16 rounding at accumulation.
    single_sample_model = deepcopy(model)
    for parameter in single_sample_model.parameters():
        parameter.data = parameter.detach().bfloat16()
    gradients = {name: [] for name, _ in single_sample_model.named_parameters()}
    denominator = torch.tensor(7, dtype=torch.float32)
    for sample in precision_inputs()[0]["samples"]:
        tokens = torch.tensor([sample["tokens"]])
        mask = torch.tensor(sample["loss_mask"][1:], dtype=torch.float32)
        single_sample_model.zero_grad(set_to_none=True)
        logits = single_sample_model(tokens).logits
        nll = torch.nn.functional.cross_entropy(logits[0, :-1].float(), tokens[0, 1:], reduction="none")
        objective = (nll * mask).sum()
        (objective / denominator if timing == "before_backward" else objective).backward()
        for name, parameter in single_sample_model.named_parameters():
            gradients[name].append(parameter.grad.detach().clone())
    for name, values in gradients.items():
        expected_gradient = torch.stack(values).float().sum(0)
        if timing == "after_backward":
            expected_gradient *= denominator.reciprocal()
        torch.testing.assert_close(ctx.tensors[0, "gradient", name], expected_gradient, rtol=0, atol=0)
    assert len(ctx.steps) == 3
    precision = ctx.metadata["reference_precision"]
    assert precision["buffer_dtypes"] == {"frequency": "torch.float32"}
    assert "does not reproduce Transformer Engine fused wgrad" in precision["limitation"]
    assert torch.equal(loaded[0].frequency, model.frequency)
    assert loaded[0].frequency.dtype == torch.float32
    assert not torch.equal(model.frequency, model.frequency.bfloat16().float())
    optimizer = ctx.suite["training"]["optimizer"]
    beta1, beta2 = optimizer["betas"]
    has_unrounded_master = False
    for name, initial in model.named_parameters():
        # Audit AdamW using FP64 recurrence on the recorded FP32 gradients;
        # this is independent of the adapter's optimizer and master storage.
        master = initial.detach().bfloat16().double()
        first_moment, second_moment = torch.zeros_like(master), torch.zeros_like(master)
        for step in range(3):
            for role in ("initial", "gradient", "parameter", "update", "exp_avg", "exp_avg_sq"):
                assert ctx.tensors[step, role, name].dtype == torch.float32
            before = ctx.tensors[step, "initial", name]
            after = ctx.tensors[step, "parameter", name]
            if step == 0:
                assert torch.equal(before, initial.detach().bfloat16().float())
            else:
                assert torch.equal(before, ctx.tensors[step - 1, "parameter", name])
            assert torch.equal(ctx.tensors[step, "update", name], after - before)
            gradient = ctx.tensors[step, "gradient", name].double()
            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
            master = master * (1 - optimizer["lr"] * optimizer["weight_decay"])
            master -= optimizer["lr"] * first_moment / (1 - beta1 ** (step + 1)) / (
                (second_moment / (1 - beta2 ** (step + 1))).sqrt() + optimizer["eps"])
            torch.testing.assert_close(ctx.tensors[step, "exp_avg", name].double(), first_moment, rtol=3e-6, atol=1e-10)
            torch.testing.assert_close(ctx.tensors[step, "exp_avg_sq", name].double(), second_moment, rtol=3e-6, atol=1e-10)
            torch.testing.assert_close(after.double(), master, rtol=3e-6, atol=3e-7)
            has_unrounded_master |= not torch.equal(after, after.bfloat16().float())
        actual_parameter = dict(loaded[0].named_parameters())[name]
        assert actual_parameter.dtype == torch.bfloat16
        assert torch.equal(actual_parameter, ctx.tensors[2, "parameter", name].bfloat16())
    assert has_unrounded_master
    assert all(ctx.tensors[step, "observation", "logits"].dtype == torch.bfloat16 for step in range(3))


@pytest.mark.parametrize("options,dtype", [
    ({"unrecognized": True}, "float32"), ({"compute_dtype": "float16"}, "float32"),
    ({"parameter_dtype": "float64"}, "float32"),
    ({"parameter_dtype": "bfloat16"}, "bfloat16"),
    ({"parameter_dtype": "bfloat16", "compute_dtype": "bfloat16"}, "float32"),
    ({"normalization_timing": "after_optimizer"}, "float32"), ({}, "float64"),
    ([], "float32"),
] + [({"micro_batch_size": value}, "float32") for value in (0, -1, 4, True, 1.5, "1")])
def test_invalid_precision_controls_fail_before_loading(monkeypatch, options, dtype):
    transformers = pytest.importorskip("transformers")

    def must_not_load(*args, **kwargs):
        pytest.fail("invalid precision controls reached checkpoint loading")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", must_not_load)
    ctx = SimpleNamespace(suite={"training": {"dtype": dtype, "global_batch_size": 3}},
                          stage={"configuration": {"options": options}})
    with pytest.raises(EvidenceError):
        run_reference(ctx)


def test_per_micro_unused_parameters_are_counted_but_whole_step_unused_is_rejected(tmp_path, monkeypatch):
    torch.manual_seed(137)
    model = TinyCausalLM(branched=True)
    ctx = MemoryReference(tmp_path / "branched", model, {"micro_batch_size": 1})
    mock_checkpoint(monkeypatch, model)
    run_reference(ctx)
    assert len(ctx.steps) == 3
    for step in ctx.steps:
        assert step["details"]["seen_micro_count"] == {"embedding.weight": 3, "head.weight": 2, "other_head.weight": 1}
    unused_model = TinyCausalLM(unused=True)
    unused_ctx = MemoryReference(tmp_path / "unused", unused_model, {"micro_batch_size": 1})
    mock_checkpoint(monkeypatch, unused_model)
    with pytest.raises(EvidenceError, match="unused trainable parameters"):
        run_reference(unused_ctx)
    assert unused_ctx.steps == []
    assert not any(role in ("parameter", "update") for _, role, _ in unused_ctx.tensors)


def test_distinct_parameters_with_shared_storage_are_rejected(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    model = TinyCausalLM()
    model.head.weight = torch.nn.Parameter(model.embedding.weight.detach())
    assert model.head.weight is not model.embedding.weight
    ctx = MemoryReference(tmp_path, model)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    with pytest.raises(EvidenceError, match="Distinct HF parameters share storage"):
        run_reference(ctx)
    assert ctx.tensors == {} and ctx.steps == []
