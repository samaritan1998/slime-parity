"""Isolated worker invoked directly or under torchrun."""
import argparse
import os
import random
import numpy as np

from .context import CaptureContext
from .model import load_symbol, symbol_source
from training_parity.common import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    context = CaptureContext(args.plan, args.stage)
    try:
        seed = context.suite["training"]["seed"]
        random.seed(seed)
        np.random.seed(seed % 2**32)
        try:
            import torch
            torch.manual_seed(seed)
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        except ImportError:
            pass
        kind = "reference" if context.stage["role"] == "reference" else "candidate"
        runner = context.stage["configuration"]["runner"]
        require(symbol_source(runner) == context.suite["sources"][kind], "Runner source changed")
        load_symbol(runner)(context)
        context.finish()
    finally:
        context.close()


if __name__ == "__main__":
    main()
