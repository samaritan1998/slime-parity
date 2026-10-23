"""Generate an editable, executable-schema suite from local model artifacts.

Slime argv, parameter semantics and numerical thresholds belong to the model
owner. No thresholds or model argument recipes are guessed by this generator.
"""
import argparse
from pathlib import Path
import random

from training_parity.common import read, write_new, require


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "parameter-map", "slime-source", "megatron-source", "slime-args", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--policy", help="JSON list of independently established numerical gates")
    args = parser.parse_args()
    require(args.steps >= 2 and args.global_batch_size > 0 and args.world_size > 0, "Invalid verification size")
    config = read(Path(args.checkpoint) / "config.json")
    vocab = config["vocab_size"]
    require(type(vocab) is int and vocab > 2, "Fixture requires a text vocabulary")
    argv = read(args.slime_args)
    require(isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv), "Slime args must be a JSON argv list")
    rng = random.Random(417)
    cases = []
    for name, length, masked in (("text", 64, False), ("masked-boundaries", 65, True)):
        batches = []
        for _ in range(args.steps):
            samples = []
            for i in range(args.global_batch_size):
                tokens = [rng.randrange(1, vocab) for _ in range(length)]
                mask = [0] + [int(not masked or (j+i) % 3 != 0) for j in range(1, length)]
                samples.append({"id": i, "tokens": tokens, "loss_mask": mask})
            batches.append({"samples": samples})
        cases.append({"id": name, "inputs": batches})
    spec = {"schema_version": 1, "name": "slime-model-parity",
            "model": {"adapter": "slime_parity.adapters.hf:HFCausalLMAdapter", "options": {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "parameter_map": str(Path(args.parameter_map).resolve())}},
            "reference": {"runner": "slime_parity.adapters.hf:run_reference", "kind": "independent",
                          "device": args.reference_device, "options": {"compute_dtype": "float32"}},
            "candidate": {"runner": "slime_parity.adapters.slime:run_slime", "kind": "slime", "device": "cuda",
                          "options": {"slime_source": str(Path(args.slime_source).resolve()),
                                      "megatron_source": str(Path(args.megatron_source).resolve()), "args": argv}},
            "training": {"steps": args.steps, "seed": 417, "global_batch_size": args.global_batch_size,
                         "dtype": args.dtype, "optimizer": {"name": "adamw", "lr": 1e-5,
                             "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0}, "clip_grad": 0},
            "cases": cases,
            "matrix": {"resolver": "megatron-dense-expert-v1", "world_size": [args.world_size],
                       "tp": [1,2], "cp": [1,2], "ep": [1,2], "sp": [False,True],
                       "pp": [1,2], "etp": [1,2], "vpp": [1,2]},
            "policy": read(args.policy) if args.policy else [],
            "execution": {"max_world_size": args.world_size, "timeout_seconds": 3600}}
    write_new(args.output, spec)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
