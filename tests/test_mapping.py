from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pytest

from slime_parity.mapping import rectangular_intervals, common_boundaries, owner_records, Interval
from training_parity.common import EvidenceError


@pytest.mark.parametrize("rows,cols", [(2,3),(3,4),(4,6)])
def test_column_shards_recover_exact_global_coordinates(rows,cols):
    full = np.arange(rows*cols).reshape(rows,cols)
    restored = np.full(full.size,-1)
    for column in range(cols):
        local = full[:,column:column+1].copy()
        for r in rectangular_intervals("w",full.shape,(0,column),local.shape):
            restored[r.start:r.end] = local.reshape(-1)[r.local_start:r.local_end]
    np.testing.assert_array_equal(restored,full.reshape(-1))


def test_transposed_factory_and_prepended_layer_axis():
    storage = np.arange(12).reshape(3,4)
    view = storage.T
    intervals = rectangular_intervals("layers.w",(8,4,3),(5,0,0),view.shape,(1,4),prepend=1)
    result = np.empty(12,dtype=int)
    for r in intervals:
        assert r.name == "layers.w::index=5"
        result[r.start:r.end] = storage.reshape(-1)[r.local_start:r.local_end]
    np.testing.assert_array_equal(result,view.reshape(-1))


def test_distributed_optimizer_range_intersection():
    original = Interval("w",(4,6),6,12,0)
    part = original.intersect(2,4)
    assert (part.start,part.end,part.local_start) == (8,10,2)
    assert original.intersect(6,10) is None


def record(rank,start,end,checksum="ok"):
    return dict(rank=rank,name="w",shape=[4],start=start,end=end,dtype="float32",sha256=checksum,fragment=0)


def test_all_replicas_checked_before_owner_election():
    catalog={"w":{"shape":[4]}}
    values=[record(0,0,2),record(1,0,2),record(2,2,4),record(3,2,4)]
    owners=owner_records(values,catalog)
    assert [v['rank'] for v in owners]==[0,2]
    bad=deepcopy(values);bad[1]['sha256']='wrong'
    with pytest.raises(EvidenceError,match="replicas disagree"):
        owner_records(bad,catalog)


@pytest.mark.parametrize("values", [[record(0,0,2)], [record(0,0,3),record(1,2,4)],
                                   [record(0,0,4),record(0,0,4)]])
def test_missing_overlap_duplicate_parameters(values):
    with pytest.raises(EvidenceError):
        owner_records(values,{"w":{"shape":[4]}})


def test_invalid_geometry_and_interval_budget():
    with pytest.raises(EvidenceError):
        rectangular_intervals("w",(4,4),(3,0),(2,4))
    with pytest.raises(EvidenceError):
        rectangular_intervals("w",(4,4),(0,0),(4,2),max_intervals=2)


def test_replica_boundaries_align_different_optimizer_partitions():
    records = [record(0, 0, 2), record(0, 2, 4),
               record(1, 0, 1), record(1, 1, 4)]
    assert common_boundaries(records, {"w": {"shape": [4]}}) == {"w": [0, 1, 2, 4]}


@pytest.mark.parametrize("fault", ["overlap", "duplicate", "gap", "shape", "elements"])
def test_replica_alignment_preserves_geometry_checks(fault):
    rows = [record(0, 0, 2), record(0, 2, 4)]
    if fault == "overlap": rows[1]["start"] = 1
    if fault == "duplicate": rows.append(deepcopy(rows[0]))
    if fault == "gap": rows[1]["start"] = 3
    if fault == "shape": rows[0]["shape"] = [5]
    if fault == "elements": rows[0]["elements"] = 1
    with pytest.raises(EvidenceError):
        common_boundaries(rows, {"w": {"shape": [4]}})


def _replica_partition_worker(rank, root, fault):
    """Real Gloo collectives over replicas whose local layouts differ."""
    import torch
    import torch.distributed as dist
    from slime_parity.mapping import write_collective

    class Writer:
        def __init__(self):
            self.rows = []

        def append(self, step, tensor, start, end, chunks):
            self.rows.append({"step": step, "tensor": tensor, "start": start,
                              "end": end, "values": np.concatenate(list(chunks)).tolist()})

    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    writer = Writer()
    try:
        value = torch.arange(8, dtype=torch.float64)
        if rank == 1 and fault == "values":
            # This element lies in a differently partitioned overlap. It must
            # be compared even though rank zero will own the emitted bytes.
            value[5] += .25
        ranges = [(0, 4), (4, 8)] if rank == 0 else [(0, 2), (2, 8)]
        if rank == 1 and fault == "overlap":
            ranges = [(0, 4), (2, 8)]
        fragments = [(Interval("w", (8,), start, end, 0), value[start:end])
                     for start, end in ranges]
        # A second tensor checks that coordinates and ownership stay separated
        # by canonical name, not only by the numeric offsets.
        fragments.append((Interval("bias", (3,), 0, 3, 0), torch.tensor([10., 20., 30.])))
        try:
            result = write_collective(writer, 2, "gradient", fragments,
                                      {"w": {"shape": [8]}, "bias": {"shape": [3]}})
            evidence = {"result": result, "rows": writer.rows}
        except EvidenceError as error:
            evidence = {"error": str(error), "rows": writer.rows}
        (root / f"rank{rank}.json").write_text(json.dumps(evidence))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", [None, "values", "overlap"])
def test_real_gloo_replica_partitions(tmp_path, fault):
    import torch.multiprocessing as mp
    mp.spawn(_replica_partition_worker, args=(str(tmp_path), fault), nprocs=2, join=True)
    evidence = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    if fault:
        expected = "replicas disagree" if fault == "values" else "within one rank"
        assert all(expected in row["error"] for row in evidence)
        assert all(not row["rows"] for row in evidence)
    else:
        assert all("error" not in row for row in evidence)
        assert evidence[1]["rows"] == []
        weights = [row for row in evidence[0]["rows"] if row["tensor"] == "gradient/w"]
        assert [(row["start"], row["end"]) for row in weights] == [(0, 2), (2, 4), (4, 8)]
        np.testing.assert_array_equal(np.concatenate([row["values"] for row in weights]), np.arange(8))
        bias = next(row for row in evidence[0]["rows"] if row["tensor"] == "gradient/bias")
        assert bias["values"] == [10., 20., 30.]
        assert all(row["result"]["replicas_checked"] == 4 for row in evidence)
