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

## Diagnose mixed-precision differences

When BF16 arithmetic is the suspected cause, use the optional
[diagnostic examples](../../examples/diagnostics/bf16/README.md) and their
[measured scope](../../docs/bf16-contract.md). Their model/version/shape limits
are local to those examples; they are not defaults for a new model's policy.

Check the actual attention backend, accumulation dtype, joint-projection
gradients and cast locations. In TP, local GEMM writeback followed by a
collective can differ from a full-matrix GEMM with one final cast. A W1-native
comparison helps localize that difference but is not an independent oracle.
Use captured operands to evaluate independently derived formulas, retaining
residuals and unobserved intermediate operations. Verify any probe against an
otherwise identical run using all canonical tensor/step bytes and dtypes.

For MoE, compare each side's own inputs, effective router weights and selected
sets at every requested step. A route mismatch can amplify small upstream
differences; local top-k consistency or a spatial association does not qualify
the full model. Do not force the reference to follow candidate routes.

The optional [NumPy semantic auditor](../../docs/moe-routing.md) checks observed
projection, selection, dispatch and combination. Validate sealed coverage,
expert identities and probe noninterference before constructing its inputs.
Use actual dispatch/scatter indices; never manufacture them from expected
routes. Missing softmax or upstream error budgets remain inconclusive, even
when routes match. Already-weighted expert outputs cannot establish internal
probability multiplication. The example W1 capture adapter supports only its
documented synthetic fixture and versions, not arbitrary models or distributed
ownership. Offline corrupted observations test the auditor; acceptance also
needs actual training fault controls.

Recompute masked shifted loss from each side's own logits; check backward
normalization separately. Identify the actual optimizer implementation and
replay its moments/update from that side's own captured gradients and previous
state. Scalar preparation and fused operation order can differ even when
nominal AdamW hyperparameters match.

Calibrate only after defining the relevant arithmetic contracts and healthy
controls. Separate calibration data from held-out controls, fix decision rules
before inspecting held-out results, and test detection of relevant injected
faults. Generation-only fixtures, repeated identical runs and smaller residuals
do not establish a qualified tolerance. Preserve the original failed matrix
and create a new frozen experiment for a changed reference or policy.

Report tested versions, cases, steps, configurations, tolerance scope and gaps.
Keep CPU observer/geometry tests separate from actual Slime GPU execution.
The built-in reference/runner is currently fresh text SFT with specified
restrictions; extensions for another objective must not silently drop checks.
Do not claim correctness outside the frozen domain or infer public release from
a successful local package build.
