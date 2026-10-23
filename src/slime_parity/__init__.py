"""Numerical correctness suites for Slime models and parallel configurations."""
from training_parity.common import EvidenceError

__version__ = "0.1.0.dev2"


def prepare(spec, output):
    from .suite import prepare as implementation
    return implementation(spec, output)


def validate(spec, output):
    path = prepare(spec, output)
    return run(path)


def run(suite_path):
    from .suite import execute
    return execute(suite_path)


def report(path):
    from .suite import report as implementation
    return implementation(path)


__all__ = ["EvidenceError", "prepare", "validate", "run", "report"]
