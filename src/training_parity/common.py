"""Portable strict JSON, content hashes, and bounded artifact paths."""
import hashlib
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    """A frozen contract or its evidence is missing, invalid, or inconsistent."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise EvidenceError(f"Not finite JSON data: {error}") from error


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    def bad_constant(value):
        raise EvidenceError(f"Nonfinite JSON constant: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=_pairs, parse_constant=bad_constant)
    canonical(value)  # Also rejects exponent overflow such as 1e999.
    return value


def write_new(path, value):
    """Never overwrite an existing experiment artifact."""
    canonical(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    name = Path(relative)
    require(not name.is_absolute() and name.parts and ".." not in name.parts,
            "Artifact path must be relative and stay inside its stage")
    current = root
    for part in name.parts:
        current /= part
        require(not current.is_symlink(), "Symlinked evidence is unsupported")
    require(current.resolve().is_relative_to(root), "Artifact escapes its root")
    return current


def shape_size(shape):
    require(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape),
            "Shape must be a list of positive dimensions; [] denotes a scalar")
    return math.prod(shape)
