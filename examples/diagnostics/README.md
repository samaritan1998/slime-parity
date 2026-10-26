# Qwen3 RoPE arithmetic diagnostic

For broader forward/backward rounding, joint projections, MoE routing and
independent AdamW replay, see the [BF16 arithmetic diagnostics](bf16/README.md).
Those optional controls also remain unqualified development references.

`qwen3_rope_reference.py` changes two parts of the HF reference: default RoPE
cosine/sine coefficients remain FP32, and split-half Q/K rotation uses FP32
multiply/add before one final cast to each input's dtype. HF still supplies the
positions, frequencies, trigonometry and scaling.

This is an altered-reference diagnostic. It is neither the standard HF reference
nor a native Slime, Megatron or Transformer Engine implementation. Smaller errors
or a completed diagnostic run do not qualify the original model as correct. Keep
the original policy and failed results unchanged; do not adopt this runner as a
replacement acceptance oracle. It does not establish TE fused-kernel or backward
equivalence.

## Pinned scope

The runner requires all of the following and rejects mismatches:

- `transformers==4.57.6`, dense `model_type="qwen3"`, and default RoPE with no
  `rope_scaling` configuration. Qwen3-MoE and other architectures are outside
  this example.
- The public HF adapter from
  [`4d4e124`](https://github.com/samaritan1998/slime-parity/commit/4d4e12422d0f0cdd1fbbeb9cbae9420defb92331),
  with `hf.py` SHA-256
  `48b96a757a280a96e256c19485ecda85a650dc6afaf5fe8928b37708df5c8cf8`.
  Review a changed adapter before updating the pin.
- BF16 training and exactly the four reference options shown below. The ordinary
  HF reference's input, mapping and optimizer restrictions still apply.

The diagnostic temporarily replaces Qwen3 class/module functions within its
worker process and restores the original attributes on normal exit or failure.
Run it in its own worker process, without concurrent Qwen3 work in that process.
Metadata records the changed arithmetic, source hashes, selected controls, call
counts and attribute restoration. The runner checks one coefficient call per
microbatch and one rotation call per layer and microbatch.

## Run a separate diagnostic suite

Use the isolated HF environment and dense fixture described in the
[synthetic Qwen3 recipe](../README.md#synthetic-qwen3-gpu-fixtures). Run these
commands from the repository root so the example and library can be imported:

```bash
export PYTHONPATH="$PWD/examples/diagnostics:$PWD/src"
cp runs/tiny-dense-fixture/spec.json runs/tiny-dense-fixture/rope-spec.json
```

Keep `training.dtype="bfloat16"`. Merge this fragment into the copied spec,
replacing its existing `reference` section; retain the checkpoint, inputs,
candidate and policy:

```json
{
  "reference": {
    "runner": "qwen3_rope_reference:run_reference",
    "kind": "independent",
    "device": "cuda",
    "options": {
      "compute_dtype": "bfloat16",
      "parameter_dtype": "bfloat16",
      "micro_batch_size": 1,
      "normalization_timing": "after_backward"
    }
  }
}
```

The schema's `kind="independent"` describes reference provenance; the altered
arithmetic remains diagnostic. Create a new plan and output directory:

```bash
slime-parity plan --spec runs/tiny-dense-fixture/rope-spec.json \
  --output runs/tiny-dense-rope-diagnostic
slime-parity run --suite runs/tiny-dense-rope-diagnostic/suite.json
```

The HF adapter keeps BF16 model parameters, separate FP32 optimizer masters and
FP32 gradient accumulators. Gradients can already have rounded in BF16 autograd
before FP32 accumulation. See [HF reference controls](../../docs/compatibility.md#hf-reference-controls)
for normalization and memory limits. Changing the runner, controls or source
requires another plan; do not modify a frozen suite or loosen its policy to
reinterpret an earlier failure.

## CPU checks

With the pinned HF version and adapter, run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_rope_diagnostic.py
```

Two layout cases check rotation values and transpose-rotation gradients using
only PyTorch. Four HF cases check FP32 identity, restoration on normal exit and
exceptions, and three optimizer steps. The HF cases explicitly skip when
Transformers is absent or its version differs from `4.57.6`; the two PyTorch
cases remain runnable. These checks do not establish GPU numerical parity.
