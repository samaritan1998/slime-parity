---
name: slime-parity-check
description: Prepare and run slime-parity model correctness suites across TP, CP, EP, SP and PP configurations, integrate a model reference, or inspect numerical and coverage failures. Use for this Python library, not ordinary training-loss analysis.
---

# Slime model parity

Use the installed `slime-parity` CLI or the supplied checkout. Keep the user's
requested model and complete parallel configuration domain. A CPU demo, a
single-axis run or an intra-Slime comparison cannot qualify the whole model.

## Prepare the experiment

Locate the model checkpoint, Slime/Megatron versions, available device capacity,
input cases, objective and numerical policy from the supplied workspace/context.
Ask only for information that materially blocks a real run; continue adapter,
planning and CPU work independently when possible. `slime-parity doctor` reports
local package versions and CUDA devices; it does not allocate resources.

For a new model, establish an independent reference and a complete canonical
trainable-parameter map before running candidates. Review QKV packing, expert
order, pipeline layers, vocabulary padding and shared weights. The catalog must
come from the independent model contract. Same-entry reference/candidate checks
prevent trivial self-comparison but do not prove semantic independence.

Use executable suite schema version 1. `slime-parity init --output spec.json`
writes a working CPU demonstration, not a real Slime model configuration.
In a source checkout, `examples/make_slime_suite.py` builds a Slime/HF suite from
local artifacts; read `docs/model-integration.zh-CN.md` for the model contract and
`docs/compatibility.md` for the current native runner boundaries.

Expand the full requested matrix. SP is boolean; dense DP and expert DP are
derived separately. Do not multiply TP, CP, EP, SP and PP as independent GPU
factors. Keep invalid/not-applicable rows and reasons. Unsupported execution and
missing resources remain unverified coverage gaps, never passed exclusions.

Freeze source, weights, inputs, steps and policy:

```bash
slime-parity plan --spec spec.json --output runs/new-plan
slime-parity report --suite runs/new-plan/suite.json
```

The report for an unexecuted plan should be unverified. Check its required
denominator before executing. Use model/device-specific thresholds established
before acceptance; no policy means report-only. Do not import CPU example
thresholds as BF16/GPU recommendations or widen a failed run's policy.

## Execute and inspect

```bash
slime-parity run --suite runs/new-plan/suite.json
```

The local runner starts only absent stages and never overwrites existing stage
directories. Use a new output directory after a failed/interrupted run or a
code/input/policy change. Preserve logs and raw evidence. A custom cluster
launcher must wait for every worker and return the actual exit status.
It is used for the reference and every candidate, including single-process
stages. An entirely invalid/inapplicable matrix retains all rows, remains
unverified and launches no reference.

Check the result in this order:

1. Requested/required/completed/passed counts and the full configuration rows.
2. Independent-reference completion, frozen artifacts, rank receipts and actual
   process-group membership. Missing/corrupted evidence cannot pass.
3. Earliest failing step/tensor and global error coordinates. Compare full
   forward/loss, reduced unscaled pre-clip gradients, actual master-parameter
   changes, resulting master parameters and Adam moments over multiple steps.
4. Model-specific semantics and input coverage, especially MoE routing and masks.

For the built-in HF/Slime SFT path, require zero dropout and rollout temperature
one. Check effective parsed precision and actual parameter dtypes; some Slime
versions force BF16, so a requested FP32 run must fail explicitly in that case.

Exit codes: 0 passed, 2 numerical failure, 3 unverified/incomplete, 4 report-only.
Global-only gates cannot qualify all required tensors. A numerical failure
identifies a discrepancy; diagnose reference, mapping, objective and native
implementation before attributing it to a kernel.

Report tested versions, cases, steps, configurations, tolerance scope and gaps.
Keep CPU observer/geometry tests separate from actual Slime GPU execution.
The built-in reference/runner is currently fresh text SFT with specified
restrictions; extensions for another objective must not silently drop checks.
Do not claim correctness outside the frozen domain or infer public release from
a successful local package build.
