# Slime Parity

[English](README.md) | [简体中文](README.zh-CN.md)

Validate a Slime model against an independent implementation across a declared
TP / CP / EP / SP / PP matrix. One model contract defines weights, inputs,
objective and canonical tensors; the suite enumerates every combination and
retains both successful results and coverage gaps.

**Development preview.** CPU execution is tested, including real two-process
Gloo and local Hugging Face runs. Native Slime execution has also completed
three-step tiny Qwen3 dense/MoE runs on one and eight GPUs, including mixed
parallelism. **Numerical acceptance still fails.** The latest development matrix
completes 112 of 128 required rows; 16 MoE rows fail execution and remain
unverified. No row passes the unchanged numerical policy. See the
[BF16 findings](docs/bf16-contract.md), [historical GPU findings](docs/gpu-validation.md)
and [current boundaries](docs/compatibility.md).
These results do not qualify all Slime models or all parallel configurations.

The [October 27–28 acceptance record (Chinese)](docs/acceptance-2026-10-27.zh-CN.md)
and [structured results](docs/acceptance-2026-10-28.json) cover the Adam counter
fix, 290 source and 290 installed-wheel CPU tests, four real GPU fault controls,
and independent recomputation of 2,151,936 report metrics. These observer and
report checks passed for the recorded scope; model numerical qualification did not.

The [follow-up controls](docs/baseline-controls.md) document explicit HF precision
settings, RoPE localization and an experimental expert-DP patch. Scoped GPU
controls confirm the ownership repair; numerical acceptance remains incomplete.

The [BF16 arithmetic record](docs/bf16-contract.md) isolates forward/backward
rounding, FA2, MoE routing and AdamW scalar preparation, with independent
diagnostic references and state-replay tools. Errors improve, while threshold
calibration and held-out acceptance remain unfinished.

The [MoE semantics API](docs/moe-routing.md) independently checks each endpoint's
router projection, top-k, actual expert dispatch and combination using NumPy.
Local results remain separate from training acceptance; missing evidence or
independent budgets stay `inconclusive`.

The [fresh-seed diagnostics](docs/cross-seed-diagnostics.md) completed 24 additional
single-GPU/mixed-layout comparisons across three seeds. All retain numerical
failures under the original policy; calibration and held-out validation remain
unfinished, and the full 128-row domain is unchanged.
Paired W1 MoE observations verify local dispatch and preweighted output sums;
probability arithmetic and cross-end stability remain unqualified.

Source: [samaritan1998/slime-parity](https://github.com/samaritan1998/slime-parity).
Install from this repository; a PyPI release is not available yet.

## Run the working CPU example

Use a Python environment with PyTorch. The core comparison package requires only
NumPy; training adapters require PyTorch, and HF references require Transformers.

```bash
git clone https://github.com/samaritan1998/slime-parity.git
cd slime-parity
python -m pip install -e '.[torch]'
slime-parity init --output cpu-suite.json
slime-parity validate --spec cpu-suite.json --output runs/cpu
```

The example executes two distinct input cases, three real AdamW updates per
case, and candidates with one and two CPU processes. The reference implements
the MLP using explicit matrix expressions; the candidate uses `torch.nn`
modules, microbatch accumulation and Gloo gradient reduction. This exercises
the library, not Slime GPU kernels. Reports are written under `runs/cpu/reports/`.

Plan and execution can also be separated:

```bash
slime-parity plan --spec cpu-suite.json --output runs/planned
slime-parity report --suite runs/planned/suite.json
slime-parity run --suite runs/planned/suite.json
slime-parity doctor
```

An unexecuted plan reports `unverified`. Existing stage directories are never
replaced, including failed attempts. Use a new output directory to retry a
failure or change code, weights, policy, cases or topology.

## Connect a Slime model

Read [the Chinese model integration guide](docs/model-integration.zh-CN.md).
The built-in `HFCausalLMAdapter` uses a local HF checkpoint and an explicit
parameter map. The candidate invokes Slime's own initialization, model builder,
HF weight loader, `train()` and `train_one_step()`. Hooks observe native
gradients and optimizer updates; the library does not substitute a training
algorithm.

```python
from slime_parity import prepare, run, report

suite_path = prepare("model-suite.json", "runs/model")
result = run(suite_path)
print(result["status"], result["counts"])
```

Each run captures, for **every declared trainable parameter at every step**:

- FP32/FP64 optimizer parameters before the update;
- reduced, unscaled gradients before clipping;
- actual optimizer parameter changes and resulting parameters;
- Adam first and second moments;
- full declared forward outputs and loss, plus input/step receipts.

Native checkpoint metadata maps TP/PP/EP and virtual pipeline chunks to global
coordinates. Distributed optimizer ranges identify owned pieces. Duplicate
replicas are split at common boundaries and must agree even when optimizer
partitions differ. Missing coverage, overlaps within a rank or disagreeing
replicas invalidate evidence.
Raw tensors stay on their owning ranks; collective communication exchanges
mapping metadata and checksums.

The matrix includes `world_size`, `tp`, `cp`, `ep`, `sp`, `pp`, `etp`, and `vpp`.
SP is a boolean; DP and expert DP are derived from the selected Megatron
convention. Invalid and inapplicable requests remain visible. Lack of resources
or an unsupported adapter path cannot shrink the required denominator.

## Read the result

| Result | Exit code | Meaning |
| --- | ---: | --- |
| `passed` | 0 | Every required case/configuration has complete evidence and passed each required tensor's numerical gates. |
| `numerical_failure` | 2 | At least one observed value exceeded a frozen threshold. Other gaps remain visible. |
| `unverified` | 3 | Some required evidence is absent, invalid, unavailable or failed to execute. |
| `report_only` | 4 | Measurements are complete, but the policy does not qualify every required tensor. |

`report.json` contains per-step, per-tensor errors, global error coordinates,
rank receipts, actual process groups, source identities and runtime versions.
`report.html` and `report.md` show the full configuration matrix. Numerical
thresholds are explicit inputs; the library never relaxes them after a failure.
See [evidence and policies](docs/evidence.md).

A pass covers the frozen test domain. It does not prove correctness for unseen
inputs, unchecked kernels, all possible configurations, convergence or model
capabilities. Reference independence and model-specific mapping semantics still
require review; hashes check consistency, not semantic correctness.

## Develop

```bash
python -m pip install -e '.[dev,hf]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

Tests cover matrix enumeration, global coordinates, corruption and incomplete
evidence, actual optimizer capture, fault injection, HF execution and Gloo DP.
Native observer tests use a small boundary harness; they do not certify
Megatron or Slime GPU execution. CPU CI is configured for Python 3.10–3.13.
See [Actions](https://github.com/samaritan1998/slime-parity/actions) for run results
and [contributing](CONTRIBUTING.md) for development guidance.

The reusable [Slime Parity Skill](skills/slime-parity-check/SKILL.md) guides model
onboarding, execution and interpretation. The original `training_parity`
package remains available for low-level tensor evidence and comparisons.

## License

Apache-2.0. See [LICENSE](LICENSE).
