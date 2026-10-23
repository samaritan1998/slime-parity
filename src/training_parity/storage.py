"""Checksummed float32/float64 shards aligned by global tensor coordinates."""
import hashlib
import math
from pathlib import Path
import numpy as np

from .common import require, read, write_new, digest, inside, shape_size
from .metrics import floating_array
from .plan import load_plan, verify_artifacts

DTYPES = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}
CHUNK_ELEMENTS = 1024 * 1024


class RankWriter:
    """Write owned intervals; adapters remain responsible for training semantics.

    Each rank has one writer for all declared steps. Different stages may use
    different rank counts and shard boundaries. No implicit float64 downcast.
    """
    def __init__(self, plan_path, stage, rank, *, max_bytes=None):
        self.plan_path = Path(plan_path).resolve()
        self.plan = load_plan(self.plan_path)
        require(stage in self.plan["stages"], "Unknown stage")
        self.stage, self.rank = stage, rank
        require(type(rank) is int and 0 <= rank < self.plan["stages"][stage]["world_size"], "Invalid rank")
        require(max_bytes is None or (type(max_bytes) is int and max_bytes > 0), "Invalid byte budget")
        self.folder = inside(self.plan_path.parent, stage)
        self.folder.mkdir(exist_ok=True)
        self.rank_dir = inside(self.folder, f"rank{rank}")
        self.rank_dir.mkdir(exist_ok=False)
        self.streams, self.hashes, self.pieces = {}, {}, []
        self.max_bytes, self.bytes, self.closed = max_bytes, 0, False

    def append(self, step, tensor, start, end, chunks):
        """Append one contiguous interval from an iterable of floating arrays."""
        require(not self.closed, "Writer is closed")
        require(type(step) is int and step in self.plan["steps"], "Unknown step")
        require(tensor in self.plan["tensors"], "Undeclared tensor")
        size = shape_size(self.plan["tensors"][tensor]["shape"])
        require(type(start) is int and type(end) is int and 0 <= start < end <= size, "Invalid global interval")
        count, dtype, offset = 0, None, None
        for chunk in chunks:
            values = floating_array(chunk).reshape(-1)
            if dtype is None:
                dtype = "float64" if values.dtype.itemsize > 4 else "float32"
                if dtype not in self.streams:
                    self.streams[dtype] = (self.rank_dir / (dtype + ".bin.partial")).open("xb")
                    self.hashes[dtype] = hashlib.sha256()
                offset = self.streams[dtype].tell()
            require(values.dtype.itemsize <= DTYPES[dtype].itemsize, "Chunk would lose precision; use a consistent storage dtype")
            require(count + values.size <= end - start, "Too many elements for interval")
            raw = values.astype(DTYPES[dtype], copy=False).tobytes()
            require(self.max_bytes is None or self.bytes + len(raw) <= self.max_bytes, "Rank byte budget exceeded")
            self.streams[dtype].write(raw)
            self.hashes[dtype].update(raw)
            self.bytes += len(raw)
            count += values.size
        require(count == end - start, "Incomplete interval")
        self.pieces.append({"step": step, "tensor": tensor, "start": start, "end": end,
                            "elements": int(count), "dtype": dtype, "offset_bytes": offset,
                            "file": f"rank{self.rank}/{dtype}.bin"})

    def write(self, step, tensor, values, *, start=0):
        """Write an array, flattened into its declared global tensor interval."""
        values = floating_array(values)
        flat = values.reshape(-1)
        self.append(step, tensor, start, start + values.size,
                    (flat[i:i + CHUNK_ELEMENTS] for i in range(0, flat.size, CHUNK_ELEMENTS)))

    def finish(self, *, observed_contract, configuration, completed_steps, metadata=None):
        """Publish the rank receipt only after native execution has completed.

        observed_contract/configuration are adapter observations, not independent
        proof. Record actual execution details and runtime versions in metadata.
        """
        require(not self.closed, "Writer is closed")
        require(observed_contract == self.plan["contract"], "Observed training identity differs from plan")
        require(configuration == self.plan["stages"][self.stage]["configuration"], "Observed stage configuration differs from plan")
        require(isinstance(completed_steps, list) and all(type(x) is int for x in completed_steps)
                and sorted(completed_steps) == self.plan["steps"], "Incomplete step execution")
        require(metadata is None or isinstance(metadata, dict), "Metadata must be an object")
        files = {}
        for dtype, stream in self.streams.items():
            stream.flush()
            stream.close()
            target = self.rank_dir / (dtype + ".bin")
            require(not target.exists(), "Raw file already exists")
            (self.rank_dir / (dtype + ".bin.partial")).rename(target)
            files[f"rank{self.rank}/{dtype}.bin"] = {
                "bytes": target.stat().st_size, "sha256": self.hashes[dtype].hexdigest()}
        receipt = {"schema_version": 1, "plan_sha256": self.plan["plan_sha256"],
                   "stage": self.stage, "rank": self.rank, "contract": observed_contract,
                   "configuration": configuration, "completed_steps": completed_steps,
                   "metadata": metadata or {}, "files": files, "pieces": self.pieces}
        write_new(self.folder / f"rank{self.rank}.json", receipt)
        self.closed = True
        return receipt

    def close(self):
        """Close partial streams without claiming a completed rank."""
        for stream in self.streams.values():
            stream.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


class Stage:
    def __init__(self, plan_path, name, *, sealed=True):
        self.plan = plan = load_plan(plan_path)
        require(name in plan["stages"], "Unknown stage")
        self.folder = folder = inside(Path(plan_path).resolve().parent, name)
        expected_ranks = {f"rank{i}.json" for i in range(plan["stages"][name]["world_size"])}
        require({p.name for p in folder.glob("rank*.json")} == expected_ranks, "Missing or extra rank receipts")
        self.ranks, self.files, self.indices = [], {}, {}
        self.rank_hashes = {n: digest(inside(folder, n)) for n in sorted(expected_ranks)}
        self.process = None
        if sealed:
            complete = read(inside(folder, "complete.json"))
            require(complete.get("plan_sha256") == plan["plan_sha256"] and complete.get("stage") == name
                    and complete.get("status") == "stage_complete", "Invalid stage completion")
            require(complete.get("ranks") == self.rank_hashes, "Rank receipt changed after completion")
            process_file = inside(folder, "process.json")
            require(digest(process_file) == complete.get("process_sha256"), "Process receipt changed")
            self.process = read(process_file)
            require(type(self.process.get("returncode")) is int and self.process["returncode"] == 0,
                    "Stage process did not exit successfully")
        for rank in range(plan["stages"][name]["world_size"]):
            row = read(inside(folder, f"rank{rank}.json"))
            require(row.get("schema_version") == 1 and row.get("plan_sha256") == plan["plan_sha256"]
                    and row.get("rank") == rank and row.get("stage") == name, "Rank identity mismatch")
            require(row.get("contract") == plan["contract"] and row.get("configuration") == plan["stages"][name]["configuration"], "Rank contract mismatch")
            require(isinstance(row.get("completed_steps"), list)
                    and all(type(s) is int for s in row["completed_steps"])
                    and sorted(row["completed_steps"]) == plan["steps"], "Rank did not complete all steps")
            require(isinstance(row.get("pieces"), list) and isinstance(row.get("files"), dict), "Invalid shard receipt")
            for piece in row["pieces"]:
                require(piece["tensor"] in plan["tensors"] and type(piece["step"]) is int and piece["step"] in plan["steps"], "Unexpected tensor or step")
                require(piece["dtype"] in DTYPES and piece["file"] in row["files"], "Unknown storage dtype or file")
                require(all(type(piece[k]) is int for k in ("start", "end", "elements", "offset_bytes")), "Non-integer interval")
                size = shape_size(plan["tensors"][piece["tensor"]]["shape"])
                require(0 <= piece["start"] < piece["end"] <= size
                        and piece["elements"] == piece["end"] - piece["start"], "Invalid interval")
                require(piece["offset_bytes"] >= 0, "Negative storage offset")
                self.indices.setdefault((piece["step"], piece["tensor"]), []).append(piece)
            for file, info in row["files"].items():
                require(file.startswith(f"rank{rank}/") and file not in self.files, "Duplicate or incorrect file ownership")
                path = inside(folder, file)
                require(path.is_file() and path.stat().st_size == info["bytes"] and digest(path) == info["sha256"], "Raw file size or checksum mismatch")
                pieces = sorted((p for p in row["pieces"] if p["file"] == file), key=lambda p: p["offset_bytes"])
                cursor = 0
                for piece in pieces:
                    require(piece["offset_bytes"] == cursor, "Gap or overlap in raw file")
                    cursor += piece["elements"] * DTYPES[piece["dtype"]].itemsize
                require(cursor > 0 and cursor == info["bytes"], "Unindexed or empty raw bytes")
                self.files[file] = info
            require(not list(inside(folder, f"rank{rank}").glob("*.partial")), "Unfinished rank data")
            self.ranks.append(row)
        expected = {(step, tensor) for step in plan["steps"] for tensor in plan["tensors"]}
        require(set(self.indices) == expected, "Missing or extra tensor/step coverage")
        for (step, tensor), pieces in self.indices.items():
            pieces.sort(key=lambda p: p["start"])
            cursor = 0
            for piece in pieces:
                require(piece["start"] == cursor, "Missing, overlapping, or duplicated global ownership")
                cursor = piece["end"]
            require(cursor == shape_size(plan["tensors"][tensor]["shape"]), "Incomplete tensor coverage")


def finalize_stage(plan_path, name, *, returncode, provenance="external_launcher", command=None):
    """Seal evidence after exit. External launchers must report the real exit code."""
    require(type(returncode) is int and returncode == 0, "Failed processes cannot complete a stage")
    require(provenance in ("external_launcher", "local_runner"), "Unknown process provenance")
    stage = Stage(plan_path, name, sealed=False)
    verify_artifacts(stage.plan)
    write_new(stage.folder / "process.json", {"returncode": returncode,
              "provenance": provenance, "command": command})
    complete = {"status": "stage_complete", "stage": name,
                "plan_sha256": stage.plan["plan_sha256"], "ranks": stage.rank_hashes,
                "process_sha256": digest(stage.folder / "process.json")}
    write_new(stage.folder / "complete.json", complete)
    return complete


def aligned_chunks(reference, candidate, key):
    left, right = reference.indices[key], candidate.indices[key]
    li = ri = cursor = 0
    while li < len(left) and ri < len(right):
        a, b = left[li], right[ri]
        end = min(a["end"], b["end"], cursor + CHUNK_ELEMENTS)
        require(end > cursor, "Invalid shard intersection")
        arrays = []
        for stage, piece in ((reference, a), (candidate, b)):
            dtype = DTYPES[piece["dtype"]]
            offset = piece["offset_bytes"] + (cursor - piece["start"]) * dtype.itemsize
            array = np.fromfile(inside(stage.folder, piece["file"]), dtype=dtype, count=end-cursor, offset=offset)
            require(array.size == end-cursor, "Truncated raw tensor")
            arrays.append(array)
        yield arrays[0], arrays[1]
        cursor = end
        li += cursor == a["end"]
        ri += cursor == b["end"]
    require(li == len(left) and ri == len(right), "Incomplete shard comparison")
