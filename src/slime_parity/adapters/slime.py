"""Native Slime SFT replay over TP/CP/EP/SP/PP, with optional VPP chunks.

Runs Slime's own train() and train_one_step(), not a replacement training loop.
The built-in IO path uses BSHD or THD text batches, native distributed Adam,
and full logits. Other model/loss layouts can register a runner using SlimeStepObserver.
GPU compatibility must be established for the selected Slime/Megatron versions.
"""
import inspect
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from training_parity.common import require, object_hash
from ..model import load_symbol
from ..mapping import MegatronMapping, Interval, torch_chunks, write_collective
from ..native import SlimeStepObserver
from ..context import input_identity
from ..topology import Topology

FLAGS = {"tp": "tensor-model-parallel-size", "cp": "context-parallel-size",
         "ep": "expert-model-parallel-size", "pp": "pipeline-model-parallel-size",
         "etp": "expert-tensor-parallel-size"}


def replace_arguments(argv, values, switches):
    """Replace known options without shell parsing or ambiguous duplicate flags."""
    require(isinstance(argv, list) and all(isinstance(v, str) and v for v in argv), "Slime arguments must be argv strings")
    result, i = [], 0
    while i < len(argv):
        item = argv[i]
        flag = item.split("=",1)[0]
        if flag in values:
            if "=" not in item:
                require(i + 1 < len(argv) and not argv[i+1].startswith("--"), f"Missing argument value: {flag}")
                i += 1
        elif flag in switches:
            require("=" not in item, f"Boolean Slime switch cannot use '=': {flag}")
        else:
            result.append(item)
        i += 1
    for key, value in values.items():
        if value is not None:
            result.extend([key, str(value)])
    result.extend(key for key,value in switches.items() if value)
    return result


def validate_native_precision(args, training, models=None):
    """Check effective defaults and parameters, not just the supplied argv."""
    import torch
    expected = getattr(torch, training["dtype"])
    require(expected in (torch.float32, torch.bfloat16), "Native IO requires FP32/BF16 parameters")
    require(not args.fp16 and bool(args.bf16) == (expected == torch.bfloat16)
            and getattr(args, "params_dtype", None) == expected,
            "Slime effective precision differs from training.dtype; this version may force BF16 defaults")
    counts = {}
    if models is not None:
        from megatron.core.utils import get_model_config
        for model in models:
            require(get_model_config(model).params_dtype == expected, "Model configuration precision differs")
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    require(parameter.dtype == expected, f"Actual model parameter precision differs: {name}")
                    key = str(parameter.dtype)
                    counts[key] = counts.get(key, 0) + parameter.numel()
    return {"requested": training["dtype"], "params_dtype": str(args.params_dtype), "parameter_elements": counts}


def validate_native_sft_objective(args):
    require(args.loss_type == "sft_loss" and args.calculate_per_token_loss, "Built-in native IO requires token-normalized SFT")
    require(getattr(args, "rollout_temperature", 1.0) == 1.0,
            "Built-in HF SFT reference requires rollout_temperature=1; register an objective adapter for scaled logits")
    for field in ("attention_dropout", "hidden_dropout", "moe_aux_loss_coeff", "moe_z_loss_coeff"):
        require(not getattr(args, field, 0), f"Built-in reference does not cover nonzero {field}; register an objective adapter")


def validate_native_compute_modes(args, models=None):
    """Parameter dtype alone does not describe quantized or replayed compute."""
    def check_configuration(config):
        values = {name: getattr(config, name, None) for name in (
            "fp8", "fp4", "fp8_param", "fp8_param_gather", "fp4_param", "use_kitchen",
            "activation_func_fp8_input_store", "moe_router_force_load_balancing")}
        values["requested_kitchen_recipe"] = any(getattr(config, name, None) is not None
                                               for name in ("kitchen_config_file", "kitchen_recipe_number"))
        values["forced_router_bias"] = getattr(config, "moe_router_force_biased", None)
        require(not any(v for k, v in values.items() if k != "forced_router_bias")
                and values["forced_router_bias"] is None,
                "Built-in reference requires unquantized FP32/BF16 compute and model-defined routing; "
                "quantized activations or forced routing needs a dedicated adapter")
        return values

    modes = check_configuration(args)
    modes["fake_int4"] = os.environ.get("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0") == "1"
    modes["routing_replay"] = os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1"
    modes["requested_routing_replay"] = bool(getattr(args, "use_routing_replay", False)
                                            or getattr(args, "use_rollout_routing_replay", False))
    require(not any(modes.values()),
            "Built-in native reference requires unquantized FP32/BF16 compute and fresh routing; "
            "fake INT4 or routing replay needs a dedicated adapter")
    if models is not None:
        from megatron.core.utils import get_model_config
        modes["models"] = []
        for chunk, model in enumerate(models):
            observed = {"chunk": chunk, **check_configuration(get_model_config(model)),
                        "module_quantization_overrides": []}
            for name, module in model.named_modules():
                params = getattr(module, "te_quant_params", None)
                recipe = getattr(params, "training_recipe", None)
                if recipe is None:
                    continue
                fp8 = getattr(recipe, "fp8_quantization_recipe", None)
                fp4 = getattr(recipe, "fp4_quantization_recipe", None)
                override = getattr(recipe, "override_nonquantized_autocast", False)
                require(not (override and (fp8 is not None or fp4 is not None)),
                        f"Native module enables quantized training outside the declared precision: {name}")
                observed["module_quantization_overrides"].append({"module": name,
                    "override_nonquantized_autocast": bool(override),
                    "fp8_recipe": None if fp8 is None else type(fp8).__name__,
                    "fp4_recipe": None if fp4 is None else type(fp4).__name__})
            modes["models"].append(observed)
    return modes


def create_session(ctx):
    import torch
    import torch.distributed as dist
    options = ctx.stage["configuration"]["options"]
    require(set(options) <= {"slime_source", "megatron_source", "args", "output_module"}, "Unknown native Slime option")
    source = Path(options["slime_source"]).resolve()
    megatron = Path(options["megatron_source"]).resolve()
    require((source / "slime/backends/megatron_utils/model.py").is_file(), "Missing Slime checkout")
    require((megatron / "megatron/training").is_dir(), "Megatron training source is required")
    sys.path[:0] = [str(source), str(megatron)]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from slime.utils.reloadable_process_group import monkey_patch_torch_dist
    monkey_patch_torch_dist()
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=ctx.suite["execution"]["timeout_seconds"]))
    from slime.utils.arguments import parse_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils import model as native_model
    from slime.backends.megatron_utils.checkpoint import _load_checkpoint_hf
    from megatron.core import parallel_state
    require(Path(inspect.getsourcefile(native_model.train)).resolve() == source / "slime/backends/megatron_utils/model.py", "Imported the wrong Slime source")
    require(Path(inspect.getsourcefile(parallel_state)).resolve() == megatron / "megatron/core/parallel_state.py", "Imported the wrong Megatron source")
    topology = Topology.from_dict(ctx.topology)
    training = ctx.suite["training"]
    layers = ctx.suite["description"]["constraints"].get("num_layers")
    require(topology.vpp == 1 or layers and layers % (topology.pp * topology.vpp) == 0, "Native built-in runner needs uniform VPP layout")
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(topology.world_size)))
    require(local_world > 0 and topology.world_size % local_world == 0, "Invalid local launcher world size")
    values = {"--"+flag: getattr(topology,axis) for axis,flag in FLAGS.items()}
    values.update({"--num-layers-per-virtual-pipeline-stage": layers // (topology.pp * topology.vpp) if topology.vpp > 1 else None,
                   "--actor-num-nodes": topology.world_size // local_world,
                   "--actor-num-gpus-per-node": local_world, "--train-backend": "megatron",
                   "--micro-batch-size": 1, "--global-batch-size": training["global_batch_size"],
                   "--lr": training["optimizer"]["lr"], "--min-lr": training["optimizer"]["lr"],
                   "--lr-decay-style": "constant", "--weight-decay": training["optimizer"]["weight_decay"],
                   "--adam-beta1": training["optimizer"]["betas"][0], "--adam-beta2": training["optimizer"]["betas"][1],
                   "--adam-eps": training["optimizer"]["eps"], "--clip-grad": training["clip_grad"],
                   "--seed": training["seed"]})
    switches = {"--sequence-parallel": topology.sp, "--use-distributed-optimizer": True, "--debug-train-only": True,
                "--bf16": training["dtype"] == "bfloat16", "--fp16": False}
    require(training["dtype"] != "float64", "Slime built-in runner supports FP32/BF16 training")
    saved = sys.argv
    try:
        sys.argv = [saved[0], *replace_arguments(options["args"], values, switches)]
        args = parse_args()
    finally:
        sys.argv = saved
    require(args.world_size == dist.get_world_size(), "Native argument validation used a different world size")
    validate_native_precision(args, training)
    ctx.metadata["compute_modes"] = validate_native_compute_modes(args)
    args.rank, args.world_size, args.local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))
    validate_native_sft_objective(args)
    layout = getattr(args, "qkv_format", "thd")
    require(layout in ("bshd", "thd") and not getattr(args, "allgather_cp", False),
            "Built-in native IO requires BSHD/THD with ordinary CP")
    ctx.metadata["input_layout"] = layout
    for field in ("reset_optimizer_states", "enable_mtp_training", "use_critic", "fp16", "optimizer_cpu_offload"):
        require(not getattr(args, field, False), f"Built-in native capture needs an adapter for {field}")
    require(getattr(args, "hf_checkpoint", None), "An HF initial checkpoint is required")
    expected_checkpoint = ctx.plan["artifacts"].get("hf_checkpoint", {}).get("path")
    require(expected_checkpoint and Path(args.hf_checkpoint).resolve() == Path(expected_checkpoint), "Native and reference checkpoints differ")
    require(not getattr(args, "load", None) or Path(args.load).resolve() == Path(expected_checkpoint),
            "Fresh verification cannot silently restore a training checkpoint")
    # Slime's builder requires an initial path; only the frozen HF checkpoint
    # is allowed here. No optimizer/training state is restored.
    args.load = expected_checkpoint
    require(getattr(args, "optimizer", "adam") == "adam", "Native observer requires Adam")
    require(not getattr(args, "lr_warmup_iters", 0) and not getattr(args, "lr_warmup_fraction", None),
            "Built-in independent reference requires a constant learning rate")
    init(args)
    models, optimizer, scheduler = native_model.setup_model_and_optimizer(args)
    _load_checkpoint_hf(models, optimizer, args, expected_checkpoint)
    ctx.metadata["precision"] = validate_native_precision(args, training, models)
    ctx.metadata["compute_modes"] = validate_native_compute_modes(args, models)
    return args, models, optimizer, scheduler, native_model


def native_token_positions(length, topology, cp_rank, layout, pad_multiplier=1):
    """Positions in a one-sample microbatch; -1 denotes a padding sequence.

    THD slices the sample for CP before padding the local stream for TP. That
    trailing padding is a separate packed sequence, not part of the sample.
    """
    require(layout in ("bshd", "thd") and type(pad_multiplier) is int and pad_multiplier > 0,
            "Unsupported native padding layout")
    alignment = 2 * topology.cp if topology.cp > 1 else 1
    if layout == "bshd":
        alignment *= topology.tp
    padded = math.ceil(length / alignment) * alignment
    positions = np.arange(padded)
    if topology.cp > 1:
        pieces = np.split(positions, 2 * topology.cp)
        positions = np.concatenate([pieces[cp_rank], pieces[-cp_rank-1]])
    if layout == "thd":
        multiple = topology.tp * pad_multiplier
        extra = (-len(positions)) % multiple
        positions = np.pad(positions, (0, extra), constant_values=-1)
    return positions


class ForwardCapture:
    def __init__(self, ctx, current, module_path):
        self.ctx, self.current, self.module_path = ctx, current, module_path
        self.handles, self.maps, self.fragments, self.seen = [], [], [], {}
        self.directory = tempfile.TemporaryDirectory(prefix="forward-", dir=ctx.writer.rank_dir)

    def install(self, modules, padded_length, topology, layout="bshd", pad_multiplier=1):
        import torch
        from megatron.core import parallel_state as ps
        shape = self.ctx.suite["description"]["observations"]["logits"]["shape"]
        batch, length, vocab = shape
        tp_rank, cp_rank = ps.get_tensor_model_parallel_rank(), ps.get_context_parallel_rank()
        positions = native_token_positions(length, topology, cp_rank, layout, pad_multiplier)
        for chunk, module in enumerate(modules):
            if not getattr(module, "post_process", False):
                continue
            target = module.get_submodule(self.module_path)
            def capture(layer, incoming, output, chunk=chunk):
                value = output[0] if isinstance(output, tuple) else output
                sample = self.current[chunk]
                require(sample is not None and value.ndim == 3 and value.shape[1] == 1 and value.shape[0] == len(positions), "Unrecognized native logits layout")
                width = value.shape[-1]
                # The standard ColumnParallelLinear exposes its full output size.
                total_width = getattr(layer, "output_size", width * topology.tp)
                require(width == total_width or width * topology.tp == total_width, "Unrecognized vocabulary sharding")
                start = 0 if width == total_width else tp_rank * width
                end = min(start + width, vocab)
                require(total_width >= vocab, "Native vocabulary is smaller than reference")
                path = Path(self.directory.name) / f"{len(self.maps)}.bin"
                dtype = np.float64 if value.dtype == torch.float64 else np.float32
                array = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(value.shape))
                flat, cursor = array.reshape(-1), 0
                for part in torch_chunks(value):
                    flat[cursor:cursor+part.size] = part
                    cursor += part.size
                array.flush()
                self.maps.append(array)
                if start >= vocab:
                    return
                for local, position in enumerate(positions):
                    if position < 0 or position >= length:
                        continue
                    key = sample, int(position), start, end
                    tensor = torch.from_numpy(array[local,0,:end-start])
                    if key in self.seen:
                        require(torch.equal(self.seen[key], tensor), "Recomputed forward observation changed")
                        continue
                    self.seen[key] = tensor
                    first = (sample * length + int(position)) * vocab + start
                    self.fragments.append((Interval("logits", tuple(shape), first, first+end-start, 0), tensor))
            self.handles.append(target.register_forward_hook(capture))

    def emit(self, step):
        catalog = {"logits": self.ctx.suite["description"]["observations"]["logits"]}
        write_collective(self.ctx.writer, step, "observation", self.fragments, catalog)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.fragments.clear(); self.seen.clear()
        for array in self.maps:
            array._mmap.close()
        self.directory.cleanup()


def run_slime(ctx):
    import torch
    import torch.distributed as dist
    args, models, optimizer, scheduler, native = create_session(ctx)
    from megatron.core import parallel_state as ps
    from megatron.core.utils import unwrap_model
    from slime.backends.megatron_utils.data import DataIterator
    topology = Topology.from_dict(ctx.topology)
    model_adapter = load_symbol(ctx.suite["model"]["adapter"])(ctx.suite["model"].get("options", {}))
    modules = unwrap_model(models)
    mapping = MegatronMapping(modules, rename=getattr(model_adapter, "rename_native", None))
    require(set(ctx.suite["description"]["observations"]) == {"loss", "logits"}, "Built-in IO needs loss and full logits; register another runner for additional observations")
    batches = ctx.suite["description"]["inputs"]
    length = len(batches[0]["samples"][0]["tokens"])
    local_steps = [data["samples"][ps.get_data_parallel_rank()::topology.dp] for data in batches]
    require(all(local and all(len(s["tokens"]) == length for s in local) for local in local_steps),
            "Built-in native IO requires fixed-length text batches")
    local_samples = [sample for batch in local_steps for sample in batch]
    alignment = topology.tp * (2 * topology.cp if topology.cp > 1 else 1)
    padded = math.ceil(length / alignment) * alignment
    require(padded <= args.seq_length, "Native seq_length cannot hold this common fixture")
    current = {i: None for i in range(len(models))}
    consumed = {i: [] for i in range(len(models))}

    def iterator_for(chunk):
        class ObservedIterator(DataIterator):
            def get_next(self, keys):
                index = self.micro_batch_indices[self.offset]
                require(len(index) == 1, "Built-in native observation requires micro batch size one")
                sample = local_samples[index[0]]
                result = super().get_next(list(dict.fromkeys([*keys, "max_seq_lens"])))
                actual = {"id": sample["id"], "tokens": result["tokens"][0].detach().cpu().tolist(),
                          "loss_mask": [0, *result["loss_masks"][0].detach().cpu().tolist()]}
                require(actual == sample, "Native iterator consumed different input")
                current[chunk] = sample["id"]
                consumed[chunk].append(object_hash(actual))
                return result
        rollout = {"tokens": [torch.tensor(s["tokens"], device="cuda", dtype=torch.long) for s in local_samples],
                   "total_lengths": [length]*len(local_samples), "response_lengths": [length-1]*len(local_samples),
                   "loss_masks": [torch.tensor(s["loss_mask"][1:], device="cuda", dtype=torch.long) for s in local_samples],
                   "rollout_mask_sums": [sum(s["loss_mask"]) for s in local_samples], "max_seq_lens": [padded]*len(local_samples)}
        return ObservedIterator(rollout, [[i] for i in range(len(local_samples))])

    iterators = [iterator_for(i) for i in range(len(models))]
    original_step, observed = native.train_one_step, []

    def observed_step(*call_args, **kwargs):
        step = len(observed)
        actual_step = inspect.signature(original_step).bind(*call_args, **kwargs).arguments["step_id"]
        require(step < len(batches) and actual_step == step, "Native optimizer step order differs")
        data, local = batches[step], local_steps[step]
        for chunk in consumed:
            consumed[chunk].clear()
            current[chunk] = None
        forward = ForwardCapture(ctx, current, ctx.stage["configuration"]["options"].get("output_module", "output_layer"))
        try:
            forward.install(modules, padded, topology, ctx.metadata["input_layout"], args.data_pad_size_multiplier)
            with SlimeStepObserver(ctx, step, models, optimizer, args, mapping=mapping):
                result = original_step(*call_args, **kwargs)
            require(all(rows == [object_hash(s) for s in local] for rows in consumed.values()),
                    "Native chunk consumed missing, duplicate, or reordered samples")
            forward.emit(step)
            rows = [None]*dist.get_world_size()
            dist.all_gather_object(rows, {k: float(v) for k, v in result[0].items()})
            values = [r["loss"] for r in rows if "loss" in r]
            require(values and all(v == values[0] for v in values), "Native reduced loss replicas disagree")
            if ctx.rank == 0:
                ctx.observe(step, "loss", values[0])
            require(scheduler.num_steps == (step+1)*len(data["samples"]), "Native scheduler sample count differs")
            ctx.record_step(step, input_identity=input_identity(data), optimizer_executed=True,
                            details={"native_entry": "slime.backends.megatron_utils.model.train_one_step",
                                     "scheduler_samples": scheduler.num_steps,
                                     "consumed": {str(k): list(v) for k,v in consumed.items()}})
            observed.append(step)
            return result
        finally:
            forward.close()

    native.train_one_step = observed_step
    try:
        native.train(0, models, optimizer, scheduler, iterators,
                     [len(local) for local in local_steps], [len(data["samples"]) for data in batches])
        require(observed == list(range(len(batches))), "Slime did not execute every optimizer step")
    finally:
        native.train_one_step = original_step
        dist.destroy_process_group()
