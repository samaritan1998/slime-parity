"""Numerical evidence and comparisons without a model or cluster dependency."""
__version__ = "0.1.0.dev2"

from .common import EvidenceError
from .metrics import compare_arrays
from .plan import prepare, load_plan
from .storage import RankWriter, finalize_stage
from .compare import compare
from .runner import run_stage

__all__ = ["EvidenceError", "compare_arrays", "prepare", "load_plan",
           "RankWriter", "finalize_stage", "compare", "run_stage"]
