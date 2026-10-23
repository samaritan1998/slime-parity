"""Canonical coordinates for native checkpoint views and optimizer slices.

Maps native checkpoint metadata and multiple model chunks into the evidence
format. Model-specific semantic renaming is an explicit adapter responsibility.
"""
from dataclasses import dataclass
from bisect import bisect_left
from itertools import product
from math import prod
import hashlib

from training_parity.common import require, shape_size


@dataclass(frozen=True)
class Interval:
    name: str
    shape: tuple
    start: int
    end: int
    local_start: int

    @property
    def local_end(self):
        return self.local_start + self.end - self.start

    def intersect(self, start, end):
        left, right = max(start, self.local_start), min(end, self.local_end)
        if right <= left:
            return None
        offset = left - self.local_start
        return Interval(self.name, self.shape, self.start + offset,
                        self.start + offset + right - left, left)


def rectangular_intervals(name, shape, offset, local_shape, strides=None, storage_offset=0,
                          prepend=0, max_intervals=1000000):
    """Map strided local storage into contiguous global row-major intervals."""
    shape, offset, local_shape = tuple(shape), tuple(offset), tuple(local_shape)
    require(type(prepend) is int and prepend >= 0, "Invalid prepended dimensions")
    require(len(shape) == len(offset) == len(local_shape) + prepend, "Shard dimensions differ")
    shape_size(list(shape)); shape_size(list(local_shape))
    if strides is None:
        strides = tuple(prod(local_shape[i+1:]) for i in range(len(local_shape)))
    strides = tuple(strides)
    require(len(strides) == len(local_shape) and all(type(s) is int and s >= 0 for s in strides), "Invalid strides")
    require(type(storage_offset) is int and storage_offset >= 0, "Invalid storage offset")
    require(all(type(o) is int and 0 <= o < n for o, n in zip(offset, shape)), "Shard offset out of bounds")
    require(all(o + n <= g for o, n, g in zip(offset[prepend:], local_shape, shape[prepend:])), "Shard extent out of bounds")
    if prepend:
        name += "::index=" + ",".join(str(i) for i in offset[:prepend])
    shape, offset = shape[prepend:], offset[prepend:]
    global_strides = tuple(prod(shape[i+1:]) for i in range(len(shape)))
    suffix, block = len(local_shape), 1
    while suffix and strides[suffix-1] == global_strides[suffix-1] == block:
        suffix -= 1
        block *= local_shape[suffix]
    require(prod(local_shape[:suffix]) <= max_intervals, "Too many strided intervals; provide a compact model mapping")
    base = sum(o*s for o, s in zip(offset, global_strides))
    result = []
    for index in product(*(range(n) for n in local_shape[:suffix])):
        first = base + sum(i*s for i, s in zip(index, global_strides))
        local = storage_offset + sum(i*s for i, s in zip(index, strides))
        row = Interval(name, shape, first, first + block, local)
        if result and result[-1].end == first and result[-1].local_end == local:
            before = result.pop()
            row = Interval(name, shape, before.start, row.end, before.local_start)
        result.append(row)
    return result


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)
    else:
        yield value


class MegatronMapping:
    """Read TP/PP/EP/factory layouts from actual model checkpoint metadata."""
    def __init__(self, models, rename=None):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
        from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map
        self.parameters, self.intervals = {}, {}
        self.rename = rename or (lambda name, shape: (name, shape))
        seen = set()
        for chunk, model in enumerate(models):
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            state = model.sharded_state_dict(metadata={"singleton_local_shards": True})
            by_id = get_param_id_to_sharded_param_map(state, params.values())
            require(set(by_id) == set(range(len(params))), "Checkpoint metadata omits model parameters")
            for i, (local_name, parameter) in enumerate(params.items()):
                require(id(parameter) not in seen, "Shared parameter across local chunks needs explicit alias mapping")
                seen.add(id(parameter))
                require(parameter.is_contiguous(), "Noncontiguous native parameters need an explicit model mapping")
                descriptor = by_id[i]
                require(descriptor.data is parameter, "Checkpoint metadata refers to another tensor")
                records = leaves(descriptor.build()) if isinstance(descriptor, ShardedTensorFactory) else [descriptor]
                intervals = []
                for shard in records:
                    require(isinstance(shard, ShardedTensor) and shard.flattened_range is None,
                            "Unsupported checkpoint factory; supply a tested model mapping")
                    data = shard.data
                    require(data is not None and data.dtype == parameter.dtype
                            and data.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr(),
                            "Checkpoint factory copied data; native storage mapping is unavailable")
                    native = rectangular_intervals(shard.key, shard.global_shape, shard.global_offset,
                                shard.local_shape, data.stride(), data.storage_offset()-parameter.storage_offset(),
                                shard.prepend_axis_num)
                    for row in native:
                        name, shape = self.rename(row.name, row.shape)
                        require(tuple(shape) == row.shape, "Rename cannot transform parameter geometry")
                        intervals.append(Interval(name, row.shape, row.start, row.end, row.local_start))
                cursor = 0
                for row in sorted(intervals, key=lambda r: r.local_start):
                    require(row.local_start == cursor, f"Overlapping or missing local storage: {local_name}")
                    cursor = row.local_end
                require(cursor == parameter.numel(), f"Incomplete local parameter: {local_name}")
                key = f"chunk{chunk}/{local_name}"
                self.parameters[key], self.intervals[key] = parameter, intervals

    def catalog(self):
        result = {}
        for rows in self.intervals.values():
            for row in rows:
                item = {"shape": list(row.shape), "group": row.name.rsplit(".", 1)[0]}
                require(result.setdefault(row.name, item) == item, "Canonical shapes disagree")
        return result

    def fragments(self, values, ranges=None):
        if ranges is None:
            require(set(values) == set(self.parameters), "Missing local parameter values")
            ranges = {n: (0, self.parameters[n].numel()) for n in values}
        require(set(values) == set(ranges) and set(values) <= set(self.parameters), "Invalid optimizer ranges")
        result = []
        for name, value in values.items():
            start, end = ranges[name]
            require(0 <= start < end <= self.parameters[name].numel() and value.numel() == end-start,
                    "Optimizer fragment size differs from model slice")
            flat = value.detach().reshape(-1)
            for interval in self.intervals[name]:
                part = interval.intersect(start, end)
                if part:
                    result.append((part, flat[part.local_start-start:part.local_end-start]))
        return result


def _validate_record(row, catalog):
    name, start, end = row["name"], row["start"], row["end"]
    require(name in catalog and row["shape"] == catalog[name]["shape"], "Canonical catalog mismatch")
    require(type(start) is int and type(end) is int and 0 <= start < end <= shape_size(row["shape"]),
            "Invalid canonical interval")
    require(type(row["rank"]) is int and row["rank"] >= 0, "Invalid canonical rank")


def common_boundaries(records, catalog):
    """Validate raw shard geometry and return a common partition per tensor.

    Shared parameters can have different optimizer boundaries in different
    pipeline stages. Each rank may cover a coordinate at most once, while
    overlaps between ranks are replicas whose values must still be checked.
    Sorting is per tensor, so unrelated parameters are not repeatedly scanned.
    """
    by_name = {}
    for row in records:
        _validate_record(row, catalog)
        require(row.get("elements", row["end"] - row["start"]) == row["end"] - row["start"],
                "Captured fragment size differs from canonical interval")
        by_name.setdefault(row["name"], []).append(row)
    boundaries = {}
    for name, tensor in catalog.items():
        cursor, rank_ends, points = 0, {}, {0, shape_size(tensor["shape"])}
        for row in sorted(by_name.get(name, []), key=lambda r: (r["start"], r["end"])):
            start, end, rank = row["start"], row["end"], row["rank"]
            require(start >= rank_ends.get(rank, 0), f"Overlapping canonical intervals within one rank: {name}")
            rank_ends[rank] = end
            require(start <= cursor, f"Missing global parameter: {name}")
            cursor = max(cursor, end)
            points.update((start, end))
        require(cursor == shape_size(tensor["shape"]), f"Incomplete canonical parameter: {name}")
        boundaries[name] = sorted(points)
    return boundaries


def owner_records(records, catalog):
    """Check aligned replica intervals, then elect one complete canonical cover.

    Values must have been hashed after splitting at ``common_boundaries``;
    whole-fragment hashes cannot compare replicas with different boundaries.
    """
    owned, seen = {}, set()
    for row in records:
        _validate_record(row, catalog)
        name, start, end = row["name"], row["start"], row["end"]
        key = name, start, end
        require((key, row["rank"]) not in seen, "Duplicate interval within one rank")
        seen.add((key, row["rank"]))
        if key in owned:
            previous = owned[key]
            require((previous["dtype"], previous["sha256"]) == (row["dtype"], row["sha256"]),
                    f"Parameter replicas disagree: {name}")
            if row["rank"] < previous["rank"]:
                owned[key] = row
        else:
            owned[key] = row
    by_name = {}
    for row in owned.values():
        by_name.setdefault(row["name"], []).append(row)
    for name, tensor in catalog.items():
        cursor = 0
        for row in sorted(by_name.get(name, []), key=lambda r: r["start"]):
            require(row["start"] == cursor, f"Missing/overlapping global parameter: {name}")
            cursor = row["end"]
        require(cursor == shape_size(tensor["shape"]), f"Incomplete canonical parameter: {name}")
    return list(owned.values())


def torch_chunks(tensor, chunk_elements=1048576):
    import torch
    tensor = tensor.detach().reshape(-1)
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    for start in range(0, tensor.numel(), chunk_elements):
        yield tensor[start:start+chunk_elements].to(device="cpu", dtype=dtype).numpy()


def write_collective(writer, step, role, fragments, catalog):
    """Synchronize only fragment metadata; raw tensor bytes stay on their owner."""
    import torch.distributed as dist
    import numpy as np
    fragments = list(fragments)
    rank, geometry = dist.get_rank(), []
    for interval, value in fragments:
        geometry.append({"name": interval.name, "shape": list(interval.shape),
                         "start": interval.start, "end": interval.end,
                         "rank": rank, "elements": value.numel()})
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, geometry)
    boundaries = common_boundaries([r for rows in gathered for r in rows], catalog)

    # Search each fragment's endpoint range, then visit only the subintervals
    # it actually covers. No all-pairs overlap scan or tensor all-gather.
    records, aligned = [], []
    for interval, value in fragments:
        points = boundaries[interval.name]
        left, right = bisect_left(points, interval.start), bisect_left(points, interval.end)
        flat = value.detach().reshape(-1)
        for index in range(left, right):
            start, end = points[index], points[index + 1]
            offset = start - interval.start
            part = flat[offset:offset + end - start]
            checksum = hashlib.sha256()
            for chunk in torch_chunks(part):
                require(np.isfinite(chunk).all(), "Nonfinite captured tensor")
                checksum.update(chunk.tobytes())
            records.append({"name": interval.name, "shape": list(interval.shape),
                            "start": start, "end": end, "dtype": str(part.dtype),
                            "sha256": checksum.hexdigest(), "rank": rank, "fragment": len(aligned)})
            aligned.append((Interval(interval.name, interval.shape, start, end,
                                     interval.local_start + offset), part))
    dist.all_gather_object(gathered, records)
    owners = owner_records([r for rows in gathered for r in rows], catalog)
    for row in owners:
        if row["rank"] == rank:
            interval, value = aligned[row["fragment"]]
            writer.append(step, f"{role}/{interval.name}", interval.start, interval.end, torch_chunks(value))
    return {"replicas_checked": len(records), "owned_fragments": sum(r["rank"] == rank for r in owners)}
