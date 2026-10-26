# Examples

`cpu-suite.json` is a complete runnable fixture: two input cases, three optimizer
steps and one/two-process candidates. Run from an installed environment:

```bash
slime-parity validate --spec examples/cpu-suite.json --output runs/cpu
```

`make_slime_suite.py` generates the executable suite schema from local HF weights,
an independently reviewed parameter map and a model's existing Slime argv. It
enumerates TP/CP/EP/SP/PP/ETP/VPP, retaining invalid/inapplicable rows. It does not
invent model arguments or numerical tolerances. See `--help` and the
[integration guide](../docs/model-integration.zh-CN.md).

Neither CPU fixtures nor generated plans establish Slime GPU compatibility.

## HF reference controls

Both suite generators retain the default HF controls: FP32 model parameters and
compute, one forward pass for the full global batch, and normalization before backward.
The reference always optimizes independent FP32 masters. With
`training.dtype="bfloat16"`, the following fragment selects explicit BF16 model
parameters, one sample per microbatch and normalization after FP32 gradient
accumulation. Merge this fragment into the generated spec, replacing its existing
`reference` section:

```json
{
  "reference": {
    "runner": "slime_parity.adapters.hf:run_reference",
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

Save the edited spec as `model-suite-controls.json`, then create and run a new
plan. The output directory must be new; existing plans and stage evidence remain
frozen.

```bash
slime-parity plan --spec model-suite-controls.json --output runs/reference-controls
slime-parity run --suite runs/reference-controls/suite.json
```

`parameter_dtype="bfloat16"` requires BF16 training and compute. FP32 parameters
with BF16 compute use autocast; explicit BF16 parameters do not. Both normalization
timings use the global supervised-token denominator. BF16 autograd gradients may
round before FP32 accumulation, so this is not a reproduction of Transformer
Engine fused wgrad. Separate FP32 masters and accumulators require additional
memory even under the defaults. See [all controls and limits](../docs/compatibility.md#hf-reference-controls).
Changing these controls defines a new reference experiment; it does not qualify
or rewrite the historical GPU results below.

See the [measured controls](../docs/baseline-controls.md), the explicit
[RoPE arithmetic diagnostic](diagnostics/README.md) and the version-pinned
[expert-DP backend patch](patches/README.md) for follow-up experiments.

## Synthetic Qwen3 GPU fixtures

`make_tiny_qwen3_suite.py` reproduces the dense and MoE fixtures used in the GPU
validation run. It creates random local weights, a complete parameter map,
Slime argv and a suite; it downloads no model or data. Use `transformers==4.57.6`
for this recipe: the MoE parameter layout in 5.x is different. Install it in a
Slime environment with compatible PyTorch, Transformer Engine and FlashAttention.
The [GPU findings](../docs/gpu-validation.md) list the tested commits, public
Megatron patch and environment versions. The fixture is specific to that model
layout; changing the source versions requires reviewing the mapping again.

Use an isolated validation environment without the optional `kernels` package.
The measured HF reference disabled optional kernel discovery. If the validation
virtual environment inherits `kernels` from its Slime installation, the following
reproduces that setting for processes launched with that environment's Python:

```bash
python - <<'PY'
from pathlib import Path
import sys
import sysconfig
assert sys.prefix != sys.base_prefix, "Select the validation virtual environment first"
path = Path(sysconfig.get_path("purelib")) / "parity_disable_optional_kernels.pth"
path.write_text('import sys; sys.modules["kernels"] = None\n')
PY
python -c 'from transformers import Qwen3ForCausalLM, Qwen3MoeForCausalLM; print("HF model imports OK")'
```

Remove that `.pth` file to restore optional discovery in the same environment.
Generate and run the suite with that interpreter:

```bash
python examples/make_tiny_qwen3_suite.py \
  --kind dense --mode matrix \
  --slime-source /path/to/slime --megatron-source /path/to/Megatron-LM \
  --policy examples/policies/tiny-qwen3-diagnostic.json \
  --output runs/tiny-dense-fixture
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0
slime-parity validate --spec runs/tiny-dense-fixture/spec.json \
  --output runs/tiny-dense-validation
```

Run `--kind moe` with separate fixture and validation directories for the MoE
case. `--mode smoke` selects one GPU and the length-64 case; `--mode mixed`
selects eight GPUs with TP=2, CP=2, SP enabled and PP=2 (EP=2 for MoE).
`--mode matrix` selects all 1/2 combinations of TP/CP/EP/PP/ETP, both SP values,
eight GPUs and VPP=1. Each matrix has two cases and three optimizer steps.

The supplied policy reproduces the original strict diagnostic limits. Those
runs reported **numerical failures**; these are not calibrated BF16 acceptance
recommendations. Omitting `--policy` produces `report_only` results. Changing
precision, versions or limits creates a different experiment and does not
change an existing failed result.
