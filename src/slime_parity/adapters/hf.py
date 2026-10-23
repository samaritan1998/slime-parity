"""Independent local Hugging Face causal-LM reference with explicit mappings.

No remote model code or automatic downloads. The native Slime model uses its
own checkpoint metadata, and an independently reviewed mapping names the HF
reference parameters in that canonical space.
"""
from pathlib import Path
from copy import deepcopy

from training_parity.common import require, read, object_hash
from training_parity.plan import inventory
from ..transforms import ParameterTransform
from ..context import input_identity
from .mlp import make_optimizer


def validate_zero_dropout(model):
    """Check modules and config fields, including functional attention dropout."""
    import torch
    config_fields = {"attention_dropout", "attention_dropout_prob", "attention_probs_dropout_prob",
                     "hidden_dropout", "hidden_dropout_prob", "dropout", "dropout_rate",
                     "activation_dropout", "embedding_dropout", "emb_dropout", "embd_dropout",
                     "residual_dropout", "resid_dropout", "drop_path_rate", "stochastic_depth_prob"}

    def stochastic_field(name):
        return isinstance(name, str) and ("dropout" in name.lower() or name.lower().endswith("_pdrop")
                                         or name in {"drop_path_rate", "stochastic_depth_prob"})

    def check_fields(values, path, config=False):
        for name, value in values.items():
            label = f"{path}.{name}"
            if isinstance(value, dict):
                check_fields(value, label, config)
            elif (stochastic_field(name) and value is not None
                  and (not config or name in config_fields or name.endswith("_pdrop"))):
                require(isinstance(value, (int, float)) and value == 0,
                        f"Built-in HF reference requires zero dropout: {label}={value}")

    # Causal-LM configs may also contain unused classifier/summary-head dropout
    # defaults. Check known causal fields here and every actual module below.
    check_fields(model.config.to_dict(), "config", config=True)
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                     torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
    for name, module in model.named_modules():
        if isinstance(module, dropout_types):
            require(module.p == 0, f"Built-in HF reference requires zero dropout: module {name}.p={module.p}")
        check_fields({key:value for key,value in vars(module).items() if stochastic_field(key)}, f"module {name}")


def normalized_inputs(case, training, vocabulary):
    require(set(case) <= {"id", "inputs", "inputs_file"}, "Unknown HF case field")
    require(("inputs" in case) != ("inputs_file" in case), "Supply inputs or inputs_file")
    values = deepcopy(case["inputs"]) if "inputs" in case else read(Path(case["inputs_file"]).resolve())
    require(isinstance(values, list) and len(values) == training["steps"], "HF inputs need one batch per optimizer step")
    length = None
    for batch in values:
        require(set(batch) == {"samples"} and len(batch["samples"]) == training["global_batch_size"], "HF global batch differs")
        ids = []
        for sample in batch["samples"]:
            require(set(sample) == {"id", "tokens", "loss_mask"}, "HF sample requires id, tokens, loss_mask")
            require(type(sample["id"]) is int, "Sample IDs must be integers")
            ids.append(sample["id"])
            tokens, mask = sample["tokens"], sample["loss_mask"]
            require(isinstance(tokens, list) and len(tokens) >= 2 and all(type(t) is int and 0 <= t < vocabulary for t in tokens), "Invalid tokens")
            require(isinstance(mask, list) and len(mask) == len(tokens) and mask[0] == 0
                    and all(type(m) is int and m in (0,1) for m in mask) and sum(mask) > 0, "Invalid next-token loss mask")
            length = len(tokens) if length is None else length
            require(len(tokens) == length, "The built-in full-logits fixture uses a common length; register another IO adapter for ragged/packed cases")
        require(ids == list(range(training["global_batch_size"])), "Samples must be in canonical ID order")
    return values, length


class HFCausalLMAdapter:
    def __init__(self, options):
        require(set(options) <= {"checkpoint", "parameter_map", "native_names", "uniform_pipeline"}, "Unknown HF model option")
        self.options = options

    def describe(self, case, training):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        checkpoint = Path(self.options["checkpoint"]).resolve()
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        validate_zero_dropout(model)
        shapes = {n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad}
        expressions = read(Path(self.options["parameter_map"]).resolve())
        transform = ParameterTransform(shapes, expressions)
        inputs, length = normalized_inputs(case, training, config.vocab_size)
        artifacts = {"hf_checkpoint": str(checkpoint), "hf_parameter_map": str(Path(self.options["parameter_map"]).resolve())}
        if "native_names" in self.options:
            artifacts["native_names"] = str(Path(self.options["native_names"]).resolve())
        if "inputs_file" in case:
            artifacts["input_cases"] = str(Path(case["inputs_file"]).resolve())
        experts = next((getattr(config, key) for key in ("num_experts", "n_routed_experts", "num_local_experts")
                        if getattr(config, key, None) is not None), 0)
        return {"parameters": {n:{"shape":shape} for n,shape in transform.shapes.items()},
                "observations": {"loss": {"shape": [], "role": "loss"},
                                 "logits": {"shape": [training["global_batch_size"], length, config.vocab_size], "role": "forward"}},
                "constraints": {"num_layers": config.num_hidden_layers, "num_experts": experts,
                                "uniform_pipeline": self.options.get("uniform_pipeline", True)},
                "identity": {"input": object_hash(inputs), "initial_state": inventory(checkpoint)["sha256"],
                             "objective": {"name": "masked_next_token_cross_entropy", "normalization": "global_supervised_tokens",
                                           "optimizer": training["optimizer"], "clip_grad": training["clip_grad"]}},
                "inputs": inputs, "artifacts": artifacts}

    def rename_native(self, name, shape):
        if "native_names" not in self.options:
            return name, shape
        names = read(Path(self.options["native_names"]).resolve())
        require(name in names, f"Native parameter has no explicit semantic name: {name}")
        return names[name], shape


def run_reference(ctx):
    import torch
    from transformers import AutoModelForCausalLM
    from contextlib import nullcontext

    options = ctx.suite["model"]["options"]
    training = ctx.suite["training"]
    device = ctx.stage["configuration"]["device"]
    checkpoint = ctx.plan["artifacts"]["hf_checkpoint"]["path"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                               attn_implementation="eager", torch_dtype=torch.float32).to(device)
    validate_zero_dropout(model)
    model.train()
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    expressions = read(ctx.plan["artifacts"]["hf_parameter_map"]["path"])
    transform = ParameterTransform({n:list(p.shape) for n,p in parameters.items()}, expressions)
    require(transform.shapes == {n:t["shape"] for n,t in ctx.suite["description"]["parameters"].items()}, "HF trainable catalog changed")
    dtype = getattr(torch, training["dtype"])
    require(dtype != torch.float64, "HF adapter currently records FP32 master parameters")
    # Use the candidate's representable initial values on both sides; compute
    # precision remains explicitly selectable for an independent reference.
    if dtype == torch.bfloat16:
        with torch.no_grad():
            for p in parameters.values():
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
    optimizer = make_optimizer(parameters.values(), training)
    reference_options = ctx.stage["configuration"]["options"]
    require(set(reference_options) <= {"compute_dtype"}, "Unknown HF reference option")
    reference_dtype = reference_options.get("compute_dtype", "float32")
    require(reference_dtype in ("float32", "bfloat16"), "Unsupported HF reference compute precision")
    def emit(step, role, values):
        for name,value in transform.apply(values).items():
            ctx.write(step, role, name, value)
    for step, data in enumerate(ctx.suite["description"]["inputs"]):
        optimizer.zero_grad(set_to_none=True)
        before = {n:p.detach().cpu().clone() for n,p in parameters.items()}
        emit(step, "initial", before)
        tokens = torch.tensor([s["tokens"] for s in data["samples"]], dtype=torch.long, device=device)
        mask = torch.tensor([s["loss_mask"] for s in data["samples"]], dtype=torch.float32, device=device)
        scope = torch.autocast(device_type=device, dtype=torch.bfloat16) if reference_dtype == "bfloat16" else nullcontext()
        with scope:
            output = model(input_ids=tokens, use_cache=False).logits
            losses = torch.nn.functional.cross_entropy(output[:, :-1].float().reshape(-1, output.shape[-1]),
                                                       tokens[:, 1:].reshape(-1), reduction="none")
            loss = (losses * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        ctx.observe(step, "logits", output)
        ctx.observe(step, "loss", loss)
        loss.backward()
        require(all(p.grad is not None for p in parameters.values()), "Reference has unused trainable parameters; declare their semantics explicitly")
        emit(step, "gradient", {n:p.grad for n,p in parameters.items()})
        if training["clip_grad"]:
            torch.nn.utils.clip_grad_norm_(parameters.values(), training["clip_grad"])
        optimizer.step()
        emit(step, "parameter", parameters)
        emit(step, "update", {n:p.detach().cpu()-before[n] for n,p in parameters.items()})
        for field in ("exp_avg", "exp_avg_sq"):
            emit(step, field, {n:optimizer.state[p][field] for n,p in parameters.items()})
        require(all(int(optimizer.state[p]["step"]) == step + 1 for p in parameters.values()), "HF optimizer step counter differs")
        ctx.record_step(step, input_identity=input_identity(data), optimizer_executed=True,
                        details={"reference": "huggingface-eager", "compute_dtype": reference_dtype})
