"""Read native Megatron optimizer ownership and observe actual training steps.

No model kernels, backward schedule, loss, or optimizer formula are replaced.
Unsupported native state layouts fail explicitly instead of inventing tensors.
"""
from pathlib import Path
import tempfile

import numpy as np
from training_parity.common import require
from .mapping import MegatronMapping, write_collective, torch_chunks
from .topology import observe_megatron_groups
from .optimizer_counter import check_native_counter


def master_parameter(optimizer, parameter):
    import torch
    state = optimizer.state.get(parameter, {})
    if "master_param" not in state:
        require(parameter.dtype in (torch.float32, torch.float64), "Low precision optimizer parameter has no master")
        return parameter
    value = (optimizer.get_unscaled_state(parameter, "master_param")
             if hasattr(optimizer, "get_unscaled_state") else state["master_param"])
    if value.dtype == torch.int16:
        require(getattr(optimizer, "store_param_remainders", False) and parameter.dtype == torch.bfloat16,
                "Unrecognized master remainder encoding")
        upper = parameter.detach().cpu().view(torch.int16).to(torch.int32) << 16
        value = (upper + value.detach().cpu().to(torch.int32)).view(torch.float32)
    require(value.dtype in (torch.float32, torch.float64) and value.shape == parameter.shape,
            "Unexpected native master shape or precision")
    return value


def optimizer_shards(optimizer, mapping):
    """Distributed Adam ranges, including chained dense/expert optimizers."""
    names = {p:n for n,p in mapping.parameters.items()}
    result = {}
    for outer in getattr(optimizer, "chained_optimizers", [optimizer]):
        if getattr(outer, "is_stub_optimizer", False):
            continue
        require(hasattr(outer, "model_param_group_index_map"),
                "Native observer requires distributed optimizer range metadata")
        native = outer.optimizer
        require(not hasattr(native, "param_to_inner_param"),
                "Hybrid CPU optimizer ownership requires a dedicated native adapter")
        for p, (group, index) in outer.model_param_group_index_map.items():
            require(p in names and names[p] not in result, "Unmapped or duplicate optimizer parameter")
            interval = outer._get_model_param_range_map(p)["param"]
            actual = native.param_groups[group]["params"][index]
            require(0 <= interval.start < interval.end <= p.numel() and actual.numel() == interval.end-interval.start,
                    "Native optimizer ownership range differs")
            result[names[p]] = {"parameter": actual, "optimizer": native, "group": native.param_groups[group],
                                "range": (interval.start, interval.end)}
    return result


def check_optimizer_options(shards, expected):
    for row in shards.values():
        group = row["group"]
        for key in ("lr", "eps", "weight_decay"):
            require(key in group and float(group[key]) == expected[key], f"Actual optimizer {key} differs from the plan")
        require(list(group.get("betas", [])) == expected["betas"], "Actual Adam betas differ")
        if expected["weight_decay"]:
            native = row["optimizer"]
            mode = getattr(native, "adam_w_mode", None)
            require(mode is not None and bool(mode) == (expected["name"] == "adamw"),
                    "Cannot verify decoupled weight-decay semantics; use an explicit optimizer adapter")


class SlimeStepObserver:
    """Context manager around the unmodified Slime train_one_step call.

    All ranks must enter/exit in the same optimizer-step order. Model mapping
    covers all local virtual chunks; global catalog comes from the independent
    model adapter. Temporary pre-step masters spill to disk, not model-sized RAM.
    """
    def __init__(self, context, step, models, optimizer, args, *, mapping=None):
        from megatron.core.utils import unwrap_model
        import torch.distributed as dist
        self.ctx, self.step, self.optimizer = context, step, optimizer
        self.mapping = mapping or MegatronMapping(unwrap_model(models))
        self.catalog = context.suite["description"]["parameters"]
        catalogs = [None] * dist.get_world_size()
        dist.all_gather_object(catalogs, self.mapping.catalog())
        merged = {}
        for catalog in catalogs:
            for name, value in catalog.items():
                require(merged.setdefault(name, value["shape"]) == value["shape"], "Native catalog shapes disagree")
        require(merged == {n:t["shape"] for n,t in self.catalog.items()}, "Native trainable catalog differs from independent catalog")
        observed = observe_megatron_groups(args, models)
        require(observed["topology"] == context.topology, "Actual parallelism differs from planned configuration")
        if "parallelism" in context.metadata:
            require(context.metadata["parallelism"] == observed, "Parallelism changed during training")
        context.metadata["parallelism"] = observed
        self.shards = optimizer_shards(optimizer, self.mapping)
        check_optimizer_options(self.shards, context.suite["training"]["optimizer"])
        self.ranges = {n:r["range"] for n,r in self.shards.items()}
        self.captured, self.updated, self.calls = False, False, 0
        self.original_prepare, self.original_step = optimizer.prepare_grads, optimizer.step
        self.temporary = None

    def masters(self):
        return {n:master_parameter(r["optimizer"], r["parameter"]) for n,r in self.shards.items()}

    def emit(self, role, values):
        return write_collective(self.ctx.writer, self.step, role,
                                self.mapping.fragments(values, self.ranges), self.catalog)

    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="before-step-", dir=self.ctx.writer.rank_dir)
        self.before = {}
        try:
            masters = self.masters()
            for i, (name, value) in enumerate(masters.items()):
                path = Path(self.temporary.name) / f"{i}.bin"
                with path.open("xb") as stream:
                    for chunk in torch_chunks(value):
                        require(np.isfinite(chunk).all(), "Nonfinite initial master")
                        stream.write(chunk.tobytes())
                dtype = np.float64 if str(value.dtype) == "torch.float64" else np.float32
                self.before[name] = np.memmap(path, dtype=dtype, mode="r", shape=(value.numel(),))
            self.emit("initial", masters)
            self.optimizer.prepare_grads = self.prepare_grads
            self.optimizer.step = self.optimizer_step
            return self
        except BaseException:
            self.cleanup()
            raise

    def prepare_grads(self, *args, **kwargs):
        result = self.original_prepare(*args, **kwargs)
        require(not result, "Native optimizer reported nonfinite gradients")
        require(not self.updated, "Gradient preparation occurred after optimizer update")
        if not self.captured:
            values = {}
            for name, row in self.shards.items():
                p = row["parameter"]
                value = getattr(p, "decoupled_grad", None)
                if value is None:
                    value = p.grad
                require(value is not None, f"Missing native gradient: {name}")
                values[name] = value
            self.emit("gradient", values)
            self.captured = True
        return result

    def optimizer_step(self, *args, **kwargs):
        import torch
        require(self.calls == 0, "Native optimizer executed more than once")
        self.calls += 1
        result = self.original_step(*args, **kwargs)
        require(isinstance(result, tuple) and len(result) == 3 and bool(result[0]), "Native optimizer skipped or failed the update")
        require(self.captured, "Native optimizer did not expose prepared gradients")
        for row in self.shards.values():
            check_native_counter(row["optimizer"].state[row["parameter"]], row["group"], self.step + 1)
        masters = self.masters()
        deltas, maps = {}, []
        try:
            for i, (name, value) in enumerate(masters.items()):
                old = self.before[name]
                path = Path(self.temporary.name) / f"delta-{i}.bin"
                array = np.memmap(path, dtype=old.dtype, mode="w+", shape=old.shape)
                maps.append(array)
                cursor = 0
                for chunk in torch_chunks(value):
                    end = cursor + chunk.size
                    array[cursor:end] = chunk - old[cursor:end]
                    cursor = end
                array.flush()
                deltas[name] = torch.from_numpy(array)
            self.emit("update", deltas)
            self.emit("parameter", masters)
            for field in ("exp_avg", "exp_avg_sq"):
                values = {}
                for name, row in self.shards.items():
                    state = row["optimizer"].state[row["parameter"]]
                    require(field in state, f"Missing native {field}")
                    value = (row["optimizer"].get_unscaled_state(row["parameter"], field)
                             if hasattr(row["optimizer"], "get_unscaled_state") else state[field])
                    values[name] = value
                self.emit(field, values)
        finally:
            deltas.clear()
            for array in maps:
                array._mmap.close()
        self.updated = True
        return result

    def cleanup(self):
        for array in getattr(self, "before", {}).values():
            array._mmap.close()
        if self.temporary:
            self.temporary.cleanup()

    def __exit__(self, kind, value, traceback):
        self.optimizer.prepare_grads, self.optimizer.step = self.original_prepare, self.original_step
        self.cleanup()
        if kind is None:
            require(self.captured and self.updated and self.calls == 1, "Native step was not fully observed")
