# Evidence and acceptance

A suite freezes a model contract, cases, input identities, all requested
configurations, source files, input artifacts and numerical rules. Its catalog
comes from the independent model description, never from whichever tensors the
candidate happened to emit. A distinct runner entry is required; different
function names/hashes are necessary checks, not proof of semantic independence.

Every required parameter has `initial`, `gradient`, `update`, `parameter`,
`exp_avg` and `exp_avg_sq` captures. `initial` and `parameter` refer to optimizer
master parameters, not a separately measured post-gather BF16 replica. Gradients
are observed after reduction/unscaling and before clipping. Updates are measured
from actual before/after parameters, not recomputed from an optimizer equation.
Forwards at subsequent steps exercise the updated model. Every step also needs
loss and at least one complete declared forward observation.

Rank writers emit global flat intervals as float32 or float64, preserving
float64 values. Rank receipts identify files, checksums, coordinates, tensor
roles, observed steps and implementation metadata. The launcher seals a stage
only after a zero process exit and complete nonoverlapping tensor coverage.
Comparison checks seals and raw bytes; a missing rank or corrupted file prevents
acceptance. Source/artifact changes invalidate suite reports.

These are consistency checks on trusted adapter output. An adapter can lie about
what it executed; checksums cannot establish provenance or prove an algorithm.
Review adapter semantics and use independently implemented references.

## Policies

Rules select canonical names, roles or groups with shell-style wildcard patterns.
Each required tensor needs at least one `scope: tensor` gate to qualify the suite.
A global-only threshold is diagnostic and cannot hide an ungated local tensor.
No policy means `report_only`.

```json
[
  {
    "id": "loss-and-forward",
    "scope": "tensor",
    "select": {"roles": ["loss", "forward"]},
    "limits": {"atol": 1e-12, "rtol": 1e-10}
  },
  {
    "id": "optimizer-tensors",
    "scope": "tensor",
    "select": {"roles": ["initial", "gradient", "update", "parameter", "exp_avg", "exp_avg_sq"]},
    "limits": {"atol": 1e-12, "rtol": 1e-10}
  }
]
```

These values illustrate the float64 CPU fixture only. They are not GPU/BF16
threshold recommendations. Establish model/device/precision tolerances in a
separate calibration experiment, then freeze acceptance rules before running
the suite. Do not widen the current run's rules after inspecting its failures.

`atol` and `rtol` implement an elementwise bound
`abs(candidate-reference) <= atol + rtol*abs(reference)`. Other supported limits
are `max_abs`, `relative_l2`, `rmse`, and `cosine_min`. Reference-zero relative L2
is zero when both tensors are zero and otherwise undefined; absolute limits
remain useful. Nonfinite values and overflow fail evidence validation.
Norms use scaled sums of squares so tiny FP64 differences do not disappear when
squared. Relative L2 is computed before rounding absolute norms. A nonzero
relative error or RMSE below FP64 representability is undefined, never zero;
a policy using that undefined metric cannot pass.

The comparison core supports reference-repeat noise policies, but the high-level
suite currently has no repeat/calibration scheduler. Noise rules in a suite
without reference repeats are rejected, never silently ignored. Candidate-to-
candidate comparisons can diagnose topology differences through the low-level
API, but they do not replace the suite's independent-reference gate.

## Coverage and reruns

`requested` counts every requested case/topology tuple. `required` counts
eligible tuples under the named resolver and model declarations. `invalid` and
`not_applicable` are retained with reasons. `resource_unavailable`, pending jobs,
execution errors and incomplete evidence remain in `required`.
If every row is invalid or inapplicable, the plan and report still retain all
requested rows, with `required: 0` and status `unverified`. Execution does not
launch an unused reference; an empty eligible domain cannot qualify as a pass.

The summary is `passed` only if all required rows pass. Any observed numerical
failure takes precedence over the overall incomplete status; counts still expose
all unfinished rows. A numerical failure is a discrepancy needing diagnosis,
not automatic proof that a particular kernel is wrong.

`run` executes only stages with no existing directory. It does not reuse failed
or partial raw files. Keep the original run, fix the issue, and prepare a new
directory. Changing a source file also requires a new plan. There is no automatic
retry, threshold rewriting, artifact deletion or GPU resource allocation.
