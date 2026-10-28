# W1 MoE observation format, version 1

Entrypoints are `moe_w1_capture:run_reference` and `moe_w1_capture:run_slime`.
This is the known four-layer, four-expert, top-k-2, W1, three-step synthetic
fixture contract, with 1,428 canonical tensor/steps per endpoint and case. It is
not a generic model capture adapter. The `required_paired_canonical_tensor_steps`
field states this prerequisite; it is not a measured count or pairing verdict.
The reference requires Transformers **4.57.6**, the published `moe_fa2` profile
and exact source pins. Observed runtimes were PyTorch **2.11.0+cu129** on GPU
and **2.14.1+cpu** for portable tests. The package's broader `torch>=2.2`
installation extra does not promise this optional observer's compatibility.
The native endpoint requires the pinned Megatron AlltoAll dispatcher and
TEGroupedMLP with W1, four local experts, top-k 2, no token dropping/padding,
no permutation/router fusion, no shared experts, and no recomputation. These
checks inspect actual objects/configuration; the observer never disables a mode.

The observer adds `moe-w1-capture.json` and `moe-w1-capture.npz` under each rank
directory, authenticated through `ctx.metadata.moe_w1_capture.files`. Metadata
`moe_w1_hooks` records installation/restoration. A failed run does not emit a
successful capture manifest. Production scope is three optimizer steps, four
layers and sixteen samples, therefore 192 records for each endpoint/case.
The actual sample token positions are explicit, including the 65-token case.

Every tensor descriptor is `{array, actual_dtype, shape, storage_dtype}`. Its
`array` names a member of the NPZ archive. BF16 values are promoted losslessly
to FP32; this promotion is storage, not the observed computation dtype. There
are no object arrays. An analyzer must validate each descriptor against the
archive rather than trusting its metadata alone.

Each record has `step`, `layer` (one-based), `sample_id`, `positions`, runtime
math flags, `router`, `helpers`, `stages`, `experts`, and `combined_output`.
The frozen fixture/source receipts must bind records to the model, parameter
map, case and endpoint. Sample identity alone is not a cross-run identity.

## Router observations

Both endpoints record actual inputs, effective BF16 weights, logits, parameter
name and expert IDs. HF expert IDs are tied to the observed ModuleList and
expert parameter prefixes. Native IDs come from the actual dispatcher local
expert indices in the pinned W1 parameter-row mapping. A consumer must keep
that source/parameter-map binding; tensor dimensions alone cannot establish
cross-endpoint expert identity.

HF records `selected`, `selected_probabilities`, `effective_probabilities` and
`selection_scores` from its original routing callback. Conversion to a dense
map/probability matrix is explicitly an analyzer transformation of these
observed slots. It must reject duplicate/out-of-range slots. The sparse block's
actual FP32 linear input, weight and output are additionally observed as
`router_projection`; the effective BF16 weight descriptor is not a substitute
for checking the actual projection operand.

Native records the actual routing function's bool `routing_map` and dense
`probabilities`. It does not invent a native top-k slot order. The gating input,
weight and output dtypes, TF32/reduced-precision flags, callable source identity,
TE availability and `te_general_gemm` availability are recorded. The selected
branch name follows the pinned Python function and actual callable availability;
it is not an observation of GEMM instructions or accumulator precision.

## Actual native dispatch and recovery

`helpers` contains exactly these original calls in this order:

1. `permute` in `dispatch_preprocess`, with actual input/map/probabilities and
   all original return values, including `sorted_indices` (return index 2).
2. `sort_chunks_by_idxs` in `dispatch_postprocess`, with actual `split_sizes`,
   `sorted_idxs`, input/probabilities and outputs.
3. `sort_chunks_by_idxs` in `combine_preprocess`, with the actual reverse
   chunk order, split sizes, input and output.
4. `unpermute` in `combine_postprocess`, with actual `sorted_indices`, input,
   restore shape, routing map and output. Its probability argument is recorded
   as supplied, including `null`; it is not inferred from the router.

`stages` records actual tensor inputs/outputs of all six dispatcher methods,
including W1 token dispatch and combine. `experts[0]` records the actual
TEGroupedMLP inputs, `counts`, probabilities, outputs and local expert indices.
Expanding these actual counts against those indices identifies the grouped
expert rows. To recover token rows, the analyzer must compose the actual
permutation and actual forward chunk sort and check every intermediate boundary.
It must also use the actual reverse chunk sort before matching the unpermute
input. A W1 identity permutation must be demonstrated from these observations,
not assumed solely because the topology is W1.

`unpermute.sorted_indices` is the actual combine destination array. Each token
can occur twice; this is not a bijective inverse over unexpanded tokens. A
canonical `(token, expert)` pair numbering is an analysis coordinate only.

## Actual reference dispatch and recovery

`experts` is the original expert-call sequence. Each entry records actual
module expert ID, inputs, probability arguments, expert output, `scatter_index`,
`scatter_source`, and output after the actual scatter. A scoped
`TorchFunctionMode` observes the original `index_add_` invocation; its source
must be the identical Python tensor object returned by the observed expert.
The original operation is called exactly once and its original return object
is returned unchanged. Rows are taken from that actual scatter index, never
reconstructed from the desired router choice. Skipped experts have no invented
call or output. The analyzer can construct counts by counting actual calls/rows.

## Evidence limits and mandatory pairing

The captured expert outputs are already probability weighted in the frozen
profiles. This observer does not capture the internal weighted activation or
prove that multiplication's correctness. A combine audit can check the sum of
observed expert outputs while the internal weighting check remains inconclusive.
No numerical tolerance, acceptance gate or routing choice is changed.

The capture manifests deliberately retain `baseline_pairing_verified=false`.
Before adopting any local semantic result, a separate sealed receipt must prove
all 1428 canonical tensor/steps have identical dtype and bytes for each side and
case against its own uninstrumented baseline. No cross-endpoint byte equality is
expected by that requirement. Complete record capture is not a semantic pass,
training acceptance, calibration or held-out evaluation.

The six portable tests in `tests/test_moe_w1_capture.py` cover the three-step
reference trajectory, actual scatter identity and wrong-index retention,
exception restoration, incomplete coverage and changed source identities. The
reference tests use a smaller two-layer CPU model to test instrumentation;
that fixture does not qualify the four-layer production capture contract.
These portable checks do not fetch Megatron, read native source checkouts or
include the source-bound native dispatcher AST tests used when developing the
observer. They do not establish native GPU behavior or production pairing.

Run them from the repository root in the pinned HF environment:

```bash
PYTHONPATH="$PWD/src:$PWD/examples/diagnostics/bf16" \
  python -m pytest -q examples/diagnostics/bf16/tests/test_moe_w1_capture.py
```
