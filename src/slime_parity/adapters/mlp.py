"""Executable CPU reference/candidate fixture (not a Slime GPU certification).

Reference uses explicit matrix expressions. Candidate uses torch.nn modules
and can execute actual Gloo data parallelism with different microbatch sizes.
"""
import os
import numpy as np

from training_parity.common import object_hash, require
from ..context import input_identity
from ..torch_capture import TorchStep
from ..topology import Topology


def weights(options, training):
    rng = np.random.default_rng(training["seed"])
    width = options.get("hidden_size", 4)
    return {"layer0.weight": rng.normal(0, 0.2, (width, 3)), "layer0.bias": rng.normal(0, 0.1, width),
            "layer1.weight": rng.normal(0, 0.2, (2, width)), "layer1.bias": rng.normal(0, 0.1, 2)}


class MLPAdapter:
    def __init__(self, options):
        require(set(options) <= {"hidden_size"}, "Unknown MLP option")
        require(type(options.get("hidden_size", 4)) is int and options.get("hidden_size", 4) > 0, "Invalid MLP width")
        self.options = options

    def describe(self, case, training):
        require(set(case) <= {"id", "seed_offset"}, "Unknown MLP case option")
        rng = np.random.default_rng(training["seed"] + case.get("seed_offset", 100))
        inputs = [{"samples": [{"id": i, "x": rng.normal(size=3).tolist(), "y": rng.normal(size=2).tolist()}
                               for i in range(training["global_batch_size"])]} for _ in range(training["steps"])]
        initial = weights(self.options, training)
        return {"parameters": {n: {"shape": list(v.shape)} for n,v in initial.items()},
                "observations": {"loss": {"shape": [], "role": "loss"},
                                 "output": {"shape": [training["global_batch_size"], 2], "role": "forward"}},
                "constraints": {"num_layers": 2, "num_experts": 0}, "inputs": inputs,
                "identity": {"input": object_hash(inputs), "initial_state": object_hash({n:v.tolist() for n,v in initial.items()}),
                             "objective": {"name": "global_mean_squared_error", "optimizer": training["optimizer"],
                                           "clip_grad": training["clip_grad"]}}, "artifacts": {}}


def make_optimizer(parameters, training):
    import torch
    options = dict(training["optimizer"])
    name = options.pop("name")
    options["betas"] = tuple(options["betas"])
    return (torch.optim.AdamW if name == "adamw" else torch.optim.Adam)(
        list(parameters), **options, foreach=False, fused=False)


def run_reference(ctx):
    import torch
    require(ctx.stage["configuration"]["device"] == "cpu", "MLP fixture requires CPU")
    require(ctx.rank == 0 and ctx.stage["world_size"] == 1, "Reference must be single-process")
    training = ctx.suite["training"]
    dtype = getattr(torch, training["dtype"])
    parameters = {n: torch.nn.Parameter(torch.tensor(v, dtype=dtype))
                  for n,v in weights(ctx.suite["model"]["options"], training).items()}
    optimizer = make_optimizer(parameters.values(), training)
    for step, data in enumerate(ctx.suite["description"]["inputs"]):
        optimizer.zero_grad(set_to_none=True)
        observed = TorchStep(ctx, step, parameters, optimizer)
        x = torch.tensor([s["x"] for s in data["samples"]], dtype=dtype)
        target = torch.tensor([s["y"] for s in data["samples"]], dtype=dtype)
        hidden = torch.tanh(x @ parameters["layer0.weight"].T + parameters["layer0.bias"])
        output = hidden @ parameters["layer1.weight"].T + parameters["layer1.bias"]
        loss = ((output - target) ** 2).sum() / target.numel()
        ctx.observe(step, "output", output)
        ctx.observe(step, "loss", loss)
        loss.backward()
        observed.capture_gradients()
        if training["clip_grad"]:
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), training["clip_grad"])
        optimizer.step()
        observed.capture_update()
        ctx.record_step(step, input_identity=input_identity(data), optimizer_executed=True,
                        details={"optimizer": type(optimizer).__name__, "forward": "explicit-matrix-reference"})


class ReplicaCheckedContext:
    def __init__(self, context, distributed):
        self.context, self.distributed = context, distributed
        self.suite = context.suite

    def write(self, step, role, name, value):
        if self.distributed:
            import torch.distributed as dist
            from ..mapping import torch_chunks
            import hashlib
            checksum = hashlib.sha256()
            for chunk in torch_chunks(value):
                checksum.update(chunk.tobytes())
            checksums = [None] * dist.get_world_size()
            dist.all_gather_object(checksums, checksum.hexdigest())
            require(len(set(checksums)) == 1, "CPU DP parameter/state replicas disagree")
        if self.context.rank == 0:
            self.context.write(step, role, name, value)


def run_candidate(ctx):
    import torch
    import torch.distributed as dist
    require(ctx.stage["configuration"]["device"] == "cpu", "MLP fixture requires CPU")
    t = Topology.from_dict(ctx.topology)
    require(t.tp == t.cp == t.ep == t.pp == t.etp == t.vpp == 1 and not t.sp,
            "The CPU fixture executes DP only; use the Slime adapter for model parallelism")
    distributed = t.world_size > 1
    if distributed:
        dist.init_process_group("gloo")
        all_ranks = list(range(t.world_size))
        singleton = None
        for i in all_ranks:
            group = dist.new_group([i], backend="gloo")
            if i == ctx.rank:
                singleton = group
        own = dist.get_process_group_ranks(singleton)
        whole = dist.get_process_group_ranks(dist.group.WORLD)
    else:
        own = whole = [0]
    ctx.metadata["parallelism"] = {"rank": ctx.rank, "topology": t.as_dict(),
        "groups": {axis: whole if axis in ("dp", "edp") else own for axis in ("tp", "cp", "pp", "dp", "ep", "etp", "edp")}}
    training = ctx.suite["training"]
    dtype = getattr(torch, training["dtype"])
    options = ctx.stage["configuration"]["options"]
    require(set(options) <= {"micro_batch_size", "fault"}, "Unknown CPU fixture option")
    micro = options.get("micro_batch_size", 1)
    require(type(micro) is int and micro > 0, "Invalid micro batch size")
    width = ctx.suite["model"]["options"].get("hidden_size", 4)
    from collections import OrderedDict
    model = torch.nn.Sequential(OrderedDict([
        ("layer0", torch.nn.Linear(3, width)), ("nonlinearity", torch.nn.Tanh()),
        ("layer1", torch.nn.Linear(width, 2))])).to(dtype=dtype)
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.copy_(torch.tensor(weights(ctx.suite["model"]["options"], training)[name], dtype=dtype))
    optimizer = make_optimizer(model.parameters(), training)
    capture = ReplicaCheckedContext(ctx, distributed)
    try:
        for step, data in enumerate(ctx.suite["description"]["inputs"]):
            optimizer.zero_grad(set_to_none=True)
            observed = TorchStep(capture, step, dict(model.named_parameters()), optimizer)
            samples = data["samples"][ctx.rank::t.world_size]
            outputs, loss_value = [], torch.zeros((), dtype=dtype)
            for first in range(0, len(samples), micro):
                batch = samples[first:first+micro]
                x = torch.tensor([s["x"] for s in batch], dtype=dtype)
                y = torch.tensor([s["y"] for s in batch], dtype=dtype)
                value = model(x)
                loss = torch.nn.functional.mse_loss(value, y, reduction="sum") / (len(data["samples"]) * 2)
                loss.backward()
                loss_value += loss.detach()
                outputs += [(s["id"], v.detach().tolist()) for s,v in zip(batch,value)]
            if distributed:
                for p in model.parameters():
                    dist.all_reduce(p.grad)
                dist.all_reduce(loss_value)
                gathered = [None] * t.world_size
                dist.all_gather_object(gathered, outputs)
                outputs = [item for rank_items in gathered for item in rank_items]
            require(sorted(i for i,_ in outputs) == list(range(len(data["samples"]))), "Missing or duplicate input samples")
            if ctx.rank == 0:
                ctx.observe(step, "output", np.asarray([v for _,v in sorted(outputs)], dtype=np.float64))
                ctx.observe(step, "loss", loss_value + (1.0 if options.get("fault") == "loss" else 0.0))
            if options.get("fault") == "gradient_sign":
                model.layer1.bias.grad[0].neg_()
            observed.capture_gradients()
            if training["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["clip_grad"])
            optimizer.step()
            observed.capture_update()
            ctx.record_step(step, input_identity=input_identity(data), optimizer_executed=True,
                            details={"optimizer": type(optimizer).__name__, "samples": [s["id"] for s in samples],
                                     "micro_batch_size": micro, "backend": "gloo" if distributed else "cpu"})
    finally:
        if distributed:
            dist.destroy_process_group()


def example_spec():
    return {"schema_version": 1, "name": "mlp-reference-and-dp", "model": {"adapter": "slime_parity.adapters.mlp:MLPAdapter", "options": {}},
            "reference": {"runner": "slime_parity.adapters.mlp:run_reference", "kind": "independent", "device": "cpu"},
            "candidate": {"runner": "slime_parity.adapters.mlp:run_candidate", "kind": "torch", "device": "cpu", "options": {"micro_batch_size": 1}},
            "training": {"steps": 3, "seed": 417, "global_batch_size": 4, "dtype": "float64", "optimizer": {"name": "adamw", "lr": 0.001}},
            "cases": [{"id": "ordinary"}, {"id": "different-inputs", "seed_offset": 200}],
            "matrix": {"world_size": [1, 2], "tp": [1], "cp": [1], "ep": [1], "sp": [False], "pp": [1]},
            "policy": [{"id": "all-tensors", "scope": "tensor", "limits": {"atol": 1e-12, "rtol": 1e-10}}],
            "execution": {"max_world_size": 2, "timeout_seconds": 120}}
