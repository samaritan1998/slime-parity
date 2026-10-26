# BF16 arithmetic controls: 2026-10-26

[English](bf16-contract.md) | [简体中文](bf16-contract.zh-CN.md)

**Numerical acceptance still fails. No new acceptance thresholds have been
selected.** This campaign isolates forward/backward rounding, gradient
accumulation and AdamW scalar preparation. The optional [independent
diagnostics](../examples/diagnostics/bf16/README.md) are reusable development
tools. Calibration, held-out positive controls and fault detection remain
unfinished. The [historical matrix](gpu-validation.md), including all 128
required configurations and its failed results, is unchanged.

The [structured record](bf16-contract-2026-10-26.json) contains allowlisted
source identities, per-step metrics and failed-gate counts. Raw runtime logs,
environment paths and platform identifiers are not public deliverables.

## Measured arithmetic

The synthetic four-layer dense/MoE Qwen3 fixtures retain hidden size 128, 4
query heads, 2 KV heads, head dimension 32 and vocabulary 512. MoE has 4 experts
and top-2 routing. Each run has three AdamW steps, batch 16, microbatch 1,
learning rate `1e-5`, betas `[0.9,0.999]`, epsilon `1e-8`, and zero weight
decay/clipping/dropout. Initial parameters round to BF16; masters and
accumulated gradients are FP32. Normalization uses the global supervised
target-token count.

The runtime has PyTorch `2.11.0+cu129`, Transformers `4.57.6` and TE `2.10.0`.
Profiler kernels identify **FA2**, with installed package `2.7.4.post1`.
Eight relevant TE Python files match public `v2.10`; this does not attest the
compiled CUDA binaries to a particular source revision.

The independent formulas model BF16 linear outputs/input gradients with FP32
weight-gradient accumulation; joint QKV and gate/up input-gradient writeback;
FP32 normalization/rotary/activation intermediates; and the observed small-input
embedding backward's repeated BF16 partial writebacks. Sparse differences from
FP32 reductions, fused operations and BF16 midpoint crossings remain visible.

Attention requires additional boundaries: unnormalized exponents round before
PV; backward recomputes probabilities and uses the stored BF16 output; dS and
per-query-head GQA gradients have separate casts. See the primary
[FA2 forward](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/flash_fwd_kernel.h)
and [backward](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/flash_bwd_kernel.h)
sources. The independent one-tile helper predicts before measuring native.
For three local inputs, forward/dV agree exactly after these changes; dQ/dK
retain sparse residuals. One dK difference after GQA cancellation is **not**
one output ULP. Scope is causal D32, no dropout, one sequence of length 1–128;
this is not long-sequence or CP-backend qualification.

The actual router uses the PyTorch FP32 fallback because the pinned Megatron
optional TE GEMM import fails. Its 36 local forward/backward comparisons match
the independent FP32/writeback formula exactly. Weighted SwiGLU places FP32
route probabilities before the BF16 activation store; local output and
gate/up gradients agree, while the probability gradient retains FP32
reduction residuals.

## Three-step model controls

All rows use the same development checkpoint and historical policy. Counts
refer to tensor/step gates, not correctness rates. Gradient relative L2 here
concatenates the entire model; acceptance remains per tensor.

| Model/reference | Failed gates | Step 0 gradient relative L2 | Step 1 | Step 2 |
| --- | ---: | ---: | ---: | ---: |
| Dense legacy BF16 HF equivalent | 76/708 | 0.0103437 | 0.0102481 | 0.0101069 |
| Dense FP32 weight gradients | 77/708 | 0.0102389 | 0.0100875 | 0.0101125 |
| Dense norm/RoPE/SwiGLU boundaries | 74/708 | 0.00989530 | 0.00967964 | 0.00970125 |
| Dense joint projections + FP32 math attention | 74/708 | 0.00883818 | 0.00860420 | 0.00862046 |
| Dense joint projections + independent FA2 boundaries | 66/708 | 0.00473667 | 0.00835322 | 0.00862096 |
| MoE legacy BF16 HF equivalent | 488/1,428 | 0.0167639 | 0.0154832 | 0.0207832 |
| MoE independent FA2/expert boundaries | 320/1,428 | 0.00835666 | 0.0119915 | 0.0155878 |

All seven dense native controls retain identical 708 tensor/step observations;
all seven MoE native controls retain byte-identical 1,428 observations.
Separate GPU runs of the exact published helpers reproduce the frozen controls:
708 dense and 1,428 MoE tensor/step observations match byte for byte on each side.
The additional masked length-65 case completes with 69/708 failed dense gates
and 308/1,428 failed MoE gates. MoE logits max-abs differences are `0.005859375`,
`0.064453125`, `0.064453125`: the first step does not establish trajectory parity.

Repeated text runs are byte-identical on both sides, including signed zeros.
Independent FP64 shifted cross-entropy recomputation covers 576 sample losses
and 36 global losses using each side's own logits and consumed tokens/masks.
Supervised denominators are 1,008 for text and 683 for the masked case; the
largest captured-loss discrepancy is `8.032e-7`. Input identities and consumed
records agree. This checks objective/normalization semantics without setting
a new tolerance.
Agreement of this forward scalar does not by itself verify backward gradient normalization.

## Gradient scaling in a mixed topology

The eight-GPU TP2/CP2/EP2/SPon/PP2/ETP1 masked MoE probe captures FP32
buffers immediately before and after the actual `DDP.scale_gradients` call,
including dense, expert and padding regions. Three steps across eight ranks
produce 48 buffer observations and 6,729,216 elements. All 1,428 canonical
tensor/step records match the same uninstrumented matrix configuration byte
for byte.

Every actual scalar equals `FP32(1/(683*2))`, where 683 is the global
supervised target-token count and two is the CP degree. Independent FP64
multiplication of each actual input by its actual FP32 scalar, rounded to
FP32, reproduces every output byte. Actual dense/expert process groups agree
with the declared topology.

This verifies the observed scaling operation in that configuration. It does
not independently qualify the preceding backward graph, gradient collectives
or every parallel configuration, and sets no new tolerance.

## Gradient reduction in the mixed topology

A separate reduction probe in the same eight-GPU masked MoE configuration
preserves all 1,428 canonical tensor/step byte sequences against the
uninstrumented matrix run. Complete dense/expert bucket and parameter-interval
records cover eight ranks and three steps: 24 bucket/step/peer-group
observations and 48 locally owned output shards. The count of 24 does not mean
24 distinct NCCL communicators.

Every captured reduce-scatter uses two participants, FP32 and SUM. Independent
FP64 addition of the two actual FP32 inputs, followed by FP32 rounding,
reproduces all 3,364,608 owned output elements byte for byte, in both parameter
and padding regions. Pre-collective buffer scaling and replay from the original
pre-sync inputs also agree exactly; byte comparisons retain signed zero.

The observed boundary is the Slime communication wrapper called by Megatron,
with output captured after the original synchronous coalescing call returns.
The wrapper source matches its frozen plan artifact; its underlying closure
targets and runtime c10d function source were not captured directly. This does
not establish NCCL internal instructions, other group sizes, asynchronous or
AVG modes, or the complete backward graph. It selects no new tolerance.

## First dense difference

A 75-site, 16-sample probe preserves all 708 training observations on both
sides. At step 0, every observed site and actual Q/K/V before the first
attention core agrees. Only 18 of its 131,072 output elements differ, with
max abs `0.00048828125` and relative L2 `2.45401e-5`. CPU evaluation of the
same independent formula on captured Q/K/V reproduces every GPU reference
output. The error propagates to logits relative L2 `0.00239559`.

Sixteen differences lie in three rows sensitive to adjacent BF16 exponent
rounding boundaries. Another is sensitive to PV accumulation; one remains
unassigned to a specific instruction. Counterfactual sensitivity does not
observe native internal exponents/reductions or establish a kernel proof.

## TP2 row-projection rounding

A paired W2/TP2/CP1/SPoff/PP1 dense probe preserves all 708 canonical
tensor/step byte sequences. The two TP ranks return identical outputs. Captured
BF16 local inputs and weights let CPU FP64 formulas compare two different
rounding contracts on the same operands, at step 0 in the first layer:

| Projection | Elements | Differences after summing ideal partial products, then casting to BF16 | Differences after casting each ideal partial to BF16, then summing and casting |
| --- | ---: | ---: | ---: |
| Attention output | 131,072 | 48,779 | 1 |
| MLP output | 131,072 | 48,943 | 8 |

The local-BF16 formula leaves max-abs residuals `1.90735e-6` and `3.05176e-5`.
Adding an FP32 round before each partial's BF16 cast changes the remaining MLP
count to five. Actual module flags confirm BF16 row projections without bias,
FP8, Userbuffers overlap or symmetric all-reduce. This is consistent with the
public [TE GEMM/writeback and collective sequence](https://github.com/NVIDIA/TransformerEngine/blob/v2.10/transformer_engine/pytorch/module/linear.py#L320).
The probe does not capture native GEMM accumulators or pre-allreduce partials;
it does not establish their instruction order, all collective implementations
or an error budget for the full parallel model.

## MoE routing and error propagation

An all-step MoE probe preserves each side's 1,428 training observations byte for
byte. It observes 12,288 layer-token positions per side, with 4,096 per step.
Every selected set agrees with that side's actual FP32 gate logits and with an
independent FP64 projection of its own captured inputs and effective weights;
there are no exact cutoff ties or FP64 top-2 inversions in these observations.

| Step | Different top-2 sets | By layer | Final-logit max abs | Potentially exposed output tokens | Fraction of squared logit error in those tokens |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 3 | `[0,0,2,1]` | 0.05859375 | 54/1,024 | 79.75% |
| 1 | 11 | `[2,1,6,2]` | 0.05322265625 | 311/1,024 | 64.20% |
| 2 | 16 | `[3,5,5,3]` | 0.05859375 | 322/1,024 | 73.35% |

The first-step legacy BF16 control had 18 route differences. In the FA2 control,
tokens outside the potentially exposed sets have max-abs logit differences
`0.005859375`, `0.0078125`, and `0.009521484375`. The exposure calculation follows
causal dependencies; it establishes a location association, not the fraction of
error caused by routing. All three full-logit maxima exceed the original `0.025`
gate. Correct local top-2 selection does not qualify the preceding activations
or the full training trajectory. The reference never adopts candidate routes.

## Independent AdamW replay

The NumPy replay uses each side's own gradients and previous captured
masters/moments. It checks the captured update identity and cross-step master
continuity, and reports residuals against independent FP64/FP32 formulas.

The all-step probe records the actual optimizer chain as `ChainedOptimizer` →
`DistributedOptimizer` → `transformer_engine.pytorch.optimizers.FusedAdam`, with
the loaded extension and base-library identities. This establishes the class
used in that run, without certifying compiled binaries against an upstream tag.

[TE FusedAdam source](https://github.com/NVIDIA/TransformerEngine/blob/v2.10/transformer_engine/common/multi_tensor/adam.cu#L541)
prepares beta as FP32 before `1-beta`: beta2 becomes
`0.9990000128746033`, with complement `0.0009999871253967285`.
HF/Python prepares scalars differently. Changing only scalar preparation in the
replay reduces MoE native second-moment max residual from `9.73110e-12` to
`2.77275e-14`, and stored parameter/update residual from `5.96046e-8` to
`3.72529e-9`; HF moment residuals worsen. This supports the scalar-rounding
explanation without asserting that every individual update improves.

These formulas do not emulate all CUDA FMA, approximate math or final fused
subtraction. They infer no tolerance. `complete_report_only` denotes a completed
replay, not numerical acceptance. See [replay instructions and
scope](../examples/diagnostics/bf16/README.md#replay-adamw-from-captured-state).

## First-step update sensitivity

A CPU analysis covers every parameter element at step 0 of the two single-GPU
(W1), length-64 FA2 text controls. Gradient bins relative to epsilon, sign classes and formulas were
fixed before this diagnostic opened the arrays. Earlier development results
already existed; this is not a blind or held-out evaluation. The table counts
only the original `abs(update_native - update_reference) > 3e-6` criterion,
not the complete policy decision.

| Model | Parameter elements | Above update absolute limit | Both gradients nonzero, opposite signs | Exactly one gradient zero | Both gradients nonzero, same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 722,304 | 946 | 854 | 60 | 32 |
| MoE | 1,117,568 | 4,943 | 4,762 | 60 | 121 |

No failing element has both gradient magnitudes at or below epsilon (`1e-8`).
Only 97 dense and 195 MoE failing elements have even one gradient that small.
These differences cannot all be described as sub-epsilon noise. For the fresh
zero-moment, zero-decay first step, the real-arithmetic update is
`-lr*g/(abs(g)+eps)`; opposite signs with magnitudes above epsilon can produce
an update difference near `2*lr = 2e-5`.

Independent replay from each side's own gradient and initial master, followed
by FP32 storage rounding, reproduces every listed limit event with no extra
events for all three replay formulas: FP64/FP32 arithmetic with each side's
scalar preparation, and an FP64 control using nominal scalars for both. The maximum residual between
predicted and observed cross-side update differences is `3.72529e-9` in each
case. This accounts for the large first-step update gaps given the captured
gradients; it does not establish why those gradients differ, or qualify the
upstream gradients, later moments or training trajectory. All elements and
original gates remain in place.

## Matrix and remaining qualification

A new development matrix retains both original models, both input cases and
every requested topology, using the explicit independent arithmetic reference
and [experimental expert-DP patch](../examples/patches/README.md). It keeps the
historical policy and does not overwrite prior evidence. The matrix fixes W8
and VPP1, varies TP/CP/EP/PP/ETP independently over one and two, and tests SP
both off and on.

Of 256 requested rows, 64 violate generic topology constraints and 64 do not
apply to Dense, leaving 128 required rows. Each row identifies a model, input
case and topology; single-GPU diagnostic controls are outside this denominator.

| Model | Required | Execution complete | Numerical failure | Numerical pass | Execution failure, unverified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 48 | 48 | 48 | 0 | 0 |
| MoE | 80 | 64 | 64 | 0 | 16 |
| Total | 128 | 112 | 112 | 0 | 16 |

Dense rows fail 66–80 gates on text and 69–79 on the masked case; MoE rows fail
320–476 and 308–523 respectively. Completed execution means three-step
computation and evidence checks completed, not numerical correctness. The
structured record's `development_matrix` field retains all row counts.

All 16 failed-execution tracebacks hit the backend guard against MoE training
with `TP>1 && SPoff`. This classification uses sealed logs; observing that
guard does not establish that bypassing it would work. Those rows remain
required and unverified, without being excluded or promoted to passed.

Ten fresh fixtures are generated and sealed: three reserved calibration and two
reserved held-out fixtures for each model, with separate weight/token seeds.
Generation performs no forward pass, routing inspection or sample selection.
The structured record includes actual hashes. This preparation covers only the
original three-step length-64/65 recipe; none of these fixtures has been evaluated.

Formal calibration still requires frozen operator contracts/error budgets,
healthy controls on the fresh data, rules fixed before opening held-out results,
and separation of healthy controls from mapping, masking,
reduction, routing and optimizer faults. Near-zero gradients can produce Adam
update sign changes; validate gradients, moments and each side's replay instead
of simply enlarging the update limit. Unsupported SP combinations and execution
failures continue to block full-domain acceptance.
