# Implementation and validation status

The complete TP/CP/EP/SP/PP verification product remains the target. The table
distinguishes implemented features from automated test coverage.

| Component | Implemented | Test coverage |
| --- | --- | --- |
| Cartesian TP/CP/EP/SP/PP/ETP/VPP matrix | All requested rows retained; conventional dense/expert legality and derived DP/EDP | 256-row enumeration test, invalid and inapplicable cases |
| Actual process-group receipts | Membership, rank coverage, reciprocal groups, commuting axes and Cartesian traversal | Real Gloo DP and synthetic malformed-group rejection |
| Global parameter mapping | Native checkpoint metadata, factory views, optimizer subranges, multiple chunks, replica agreement across different partition boundaries | CPU geometry, permutation, ownership, missing/overlap tests and real two-rank Gloo replica checks |
| Independent reference | Explicit MLP and local HF causal LM | Three real optimizer steps in both paths |
| Native Slime execution | Original init/build/load/train; forward and optimizer observers | Three-step tiny Qwen3 dense/MoE execution on one/eight GPUs; numerical failures retained in [GPU findings](gpu-validation.md) |
| Numerical evidence | Full forward/loss/gradient/actual update/parameters/Adam moments, every step | CPU/GPU capture, local failures and artifact corruption |
| Packaging and CLI | Python library, console commands, wheel/sdist, CPU CI configuration | Build and CLI tests; hosted results in [Actions](https://github.com/samaritan1998/slime-parity/actions) |

The native runner was written against Slime's `megatron_utils` interfaces,
including the public source inspected at
[`18868a79b0b083a093798a94ed0f8cb4f71b9c14`](https://github.com/THUDM/slime/tree/18868a79b0b083a093798a94ed0f8cb4f71b9c14).
This checkout was used in the GPU run, whose numerical gates did not all pass. Each
run freezes the actual supplied checkout and rejects imports from another path.
Compatibility results apply only to the tested versions and configurations.

## Current built-in Slime/HF path

The runner accepts fresh local HF weights, ordinary text causal-LM SFT, a common
sequence length within each case, BSHD or THD input, mirrored two-chunk CP partitioning,
microbatch size one, uniform PP/VPP layouts and a native distributed Adam
optimizer. Global batch size must divide evenly across dense DP. It uses a
constant LR, zero dropout, rollout temperature one and zero MoE auxiliary/z-loss.
The reference checks both configuration fields and actual dropout modules.
The reference runs HF
eager attention with FP32 optimizer parameters; BF16 candidate initial values
are rounded on both sides. Every nonzero weight-decay group must match the
declared optimizer semantics; models with special parameter groups need a
dedicated objective adapter.

The input layout follows the selected Slime version: older versions expose
`--qkv-format`, while the pinned public checkout above uses THD without that
argument. THD capture accounts for CP slicing followed by local TP alignment;
padding sequences are excluded from the canonical logits. This does not add
support for multiple variable-length samples packed into one microbatch.

The runner verifies parsed `fp16`/`bf16` flags, `params_dtype`, model configuration
and actual trainable-parameter dtypes against the requested precision. A Slime
version that forces BF16 rejects a requested FP32 run; it cannot silently record
that run as FP32. Mixed parameter precision requires an explicit adapter.
The standard reference also rejects known FP8/FP4 compute or parameter storage,
Kitchen quantization recipes, fake-INT4 compute, FP8 saved
activations, active per-module TE quantization overrides, forced routing and
requested routing replay. Parameter dtype alone cannot establish those semantics.
Inactive quantization recipe settings do not count as enabled compute.

The following need model/version-specific extensions and remain **unverified**
when requested: RL rollout/objective replay, multimodal inputs, packed/ragged
sequences, dynamic/allgather CP, GTP, heterogeneous pipelines, MTP, CPU optimizer
offload, checkpoint resume, custom stochastic routing and nonzero auxiliary
objectives. Unsupported native metadata, copied checkpoint factory tensors,
unmapped shared-parameter aliases and unused reference parameters fail explicitly.

EP and all mixed dimensions are present in the matrix and native mapping path.
Tiny Qwen3-MoE has GPU execution evidence, including a mixed EP=2 run, but has
not passed its frozen numerical gates. With the tested backend, TP=1/EP=1/ETP=2
also failed optimizer-master coverage before training; source inspection and a
CPU reproduction point to expert parameter grouping. Those configurations
remain unverified; see the [GPU findings](gpu-validation.md).
That backend also rejects MoE training with TP>1 and SP disabled, including
some configurations that pass the planner's generic constraints. Eligible
planned rows still require native preflight and actual execution evidence.
A model adapter must
independently describe its expert order, routing objective and trainable catalog.
Cases must exercise routing sufficiently; the built-in text generator is not
evidence that every expert/route has been exercised.

## Scale and launch boundaries

- The default launcher is local `torchrun`; a custom argv launcher can integrate
  an existing scheduler. It applies to every stage, including the independent
  reference and single-process candidates, and must wait for all workers and return their actual
  exit status. It must set rank variables and expose the frozen source/artifact
  paths. Multi-node launch remains unverified.
- Parameter bytes are streamed to per-rank disk; metadata is all-gathered. Very
  fragmented layouts can require substantial metadata memory. There is an
  explicit per-shard interval bound, not a proven billion-parameter scale claim.
- The native observer spills pre-update masters and output snapshots to disk.
  The built-in HF reference holds its model/optimizer and CPU before-update
  snapshots in memory. Large references need an offload/distributed adapter;
  inability to run a reference is a coverage gap, not a pass.
- Full logits and six parameter-role captures at every step can consume large
  storage. `rank_byte_budget` bounds emitted evidence bytes, not temporary
  snapshots, kernel caches or total filesystem usage. Budget storage accordingly.
- Reports contain local artifact/source paths and runtime metadata. They have
  no automatic public-report redaction. Release artifacts should exclude private
  weights, inputs, logs and evidence unless explicitly selected for publication.

## Qualifying another model or environment

For a supported dense model and then an MoE model, pin Slime, Megatron,
Transformer Engine, PyTorch, device type and checkpoint; review the independent
parameter mapping and objective; freeze cases and thresholds. Execute the full
declared matrix, including individual axes and mixed TP+CP+EP+SP+PP layouts.
Keep unavailable, invalid and adapter-failure rows visible. Only the observed
results establish compatibility for those versions and cases.
