# Fresh-seed prerequisite diagnostics: 2026-10-28 UTC

[English](cross-seed-diagnostics.md) | [简体中文](cross-seed-diagnostics.zh-CN.md)

**All 24 additional comparisons completed; none passed the unchanged numerical policy.** No formal calibration or held-out acceptance was performed. Three new weight/token seeds from the reserved calibration partition test the earlier observations on fresh data; these runs are not certified healthy controls.

The [structured report](cross-seed-diagnostics-2026-10-28.json) retains per-seed/case/layout step metrics and failed gate counts. The historical full domain remains 128 required rows: 112 completed numerical failures and 16 unverified MoE execution failures at the backend `TP>1 && SPoff` guard. These 24 diagnostics do not replace that denominator or bypass the guard.

## Frozen scope

Weight/token seed pairs are `(1103,2111)`, `(2203,2113)` and `(3301,2129)`. Fixtures were generated and sealed before execution; every declared seed was retained without output-based selection. The two held-out fixtures per model were not evaluated.

The models are synthetic four-layer Qwen3 Dense and Qwen3-MoE, using the published independent BF16/FA2 diagnostic references. Each model has 64-token text and 65-token masked-boundary cases. Each row runs three steps with global batch 16 and microbatch 1. New seeds change the weights and token inputs; each seed/case freezes identical inputs and initial weights for both endpoints and layouts. AdamW, BF16 storage and the original per-tensor policy are unchanged. See the [earlier record](bf16-contract.md) for the arithmetic recipe.

Each seed uses two layouts:

- W1: TP/CP/EP/PP/ETP/VPP all 1, SP off.
- Eight-GPU mixed: TP2, CP2, PP2, ETP1, VPP1, SP on; EP1 for Dense and EP2 for MoE.

This gives `2 models × 3 seeds × 2 cases × 2 layouts = 24` candidate rows, plus 24 independent reference executions. Every row has complete three-step evidence and per-tensor gates. Within each comparison, the two endpoints' initial step-0 master parameters agree. Observed versions are PyTorch `2.11.0+cu129`, CUDA `12.9`, Transformers `4.57.6`, Transformer Engine `2.10.0` and FlashAttention `2.7.4.post1`.

## Observed ranges

Each group contains six rows: three seeds and two cases. Failed gate counts cover one complete three-step row, with 708 gates for Dense or 1,428 for MoE. Gradient relative L2 concatenates the complete gradient at each step. Both the gradient and logit ranges cover 18 step observations per group, not six three-step row maxima, and never replace per-tensor acceptance.

| Model/layout | Complete / required | Passed | Failed gates per row | Gradient relative L2 range | Full-logit max-absolute-error range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense, W1 | 6/6 | 0 | 60–69 / 708 | 0.00472446–0.00925535 | 0.0048828125–0.009765625 |
| Dense, eight-GPU mixed | 6/6 | 0 | 78–92 / 708 | 0.00971481–0.0108738 | 0.0087890625–0.01171875 |
| MoE, W1 | 6/6 | 0 | 301–370 / 1,428 | 0.00424855–0.0206121 | 0.005859375–0.09765625 |
| MoE, eight-GPU mixed | 6/6 | 0 | 461–567 / 1,428 | 0.0149299–0.0217054 | 0.05419921875–0.099609375 |

Residuals on the old development seed cannot alone define general tolerances, especially for an entire MoE trajectory from one first-step observation. These are observed ranges, not recommended tolerances, and were not written back into the policy.

## Actual W1 MoE boundaries

The two cases for each of the three MoE seeds also ran with the W1 observer.
The published [capture adapter](../examples/diagnostics/bf16/moe-w1-capture.md)
observes actual routing, expert calls, dispatch and inverse combination within
this tiny fixture's scope. The [structured routing summary](moe-routing-2026-10-28.json)
retains per-seed/case counts, source identities and limitations.

Each endpoint has `3 seeds × 2 cases × 3 steps × 4 layers × 16 samples = 1,152`
layer records, or 2,304 across both endpoints. Each endpoint/case was compared
with its own uninstrumented baseline over 1,428 canonical tensor/steps. All
17,136 comparisons across 12 pairs have identical dtype and original bytes,
including signed zero. CPU analysis reverified sealed identities, complete
coverage and these paired results.

| Local check | Reference | Slime |
| --- | ---: | ---: |
| Conditional FP32 projection bound on own effective BF16 operands | 1,152/1,152 pass | 1,152/1,152 pass |
| Legal top-k on own actual scores | 1,152/1,152 pass | 1,152/1,152 pass |
| Actual dispatch permutation, expert counts, input/probability transport and inverse indices | 1,152/1,152 pass | 1,152/1,152 pass |
| Sum of two already-weighted BF16 expert outputs | 1,152/1,152 pass | 1,152/1,152 pass |
| Softmax probability and probability-sum error budgets | 1,152 inconclusive | 1,152 inconclusive |
| Probability multiplication inside experts | 1,152 inconclusive | 1,152 inconclusive |

The projection result assumes ordinary FP32 multiply/add or FMA. Actual source
branches, operand dtypes, TF32 and autocast flags were checked; GPU instructions
were not captured. The predeclared zero-error sum bound checks inverse transport
and addition of observed preweighted outputs, not the expert function or its
internal probability multiplication. Overall routing and combine therefore
remain `inconclusive`.

Of 74,304 aligned step/layer/sample/token coordinates, 193 have different expert
sets across endpoints:

| Weight/token seed | 64-token text | 65-token masked boundaries |
| --- | ---: | ---: |
| 1103 / 2111 | 28 | 32 |
| 2203 / 2113 | 36 | 30 |
| 3301 / 2129 | 39 | 28 |

These counts include multiple layers and steps, not distinct input tokens.
Each endpoint selects legally from its own actual scores. No independently
justified budget covers upstream activation and weight differences, so all
1,152 paired layer records remain `inconclusive`, including matching routes.
The 193 differences alone neither establish a router defect nor qualify as
acceptable rounding error.

Even the conservative same-operand projection intervals do not establish a
unique route set everywhere: 28 reference and 23 Slime token coordinates remain
uncertain, with zero exact cutoff ties in either endpoint's actual scores.
Legal top-k membership differs from a uniquely stable set throughout an error
interval. These counts cannot directly explain the 193 cross-end differences.

Offline mutations of the sealed observations exercised wrong inverse indices,
misrouted probabilities, wrong row identities and illegal top-k sets; each fault
type was detected 2,304 times. These test the auditor, not actual GPU training
fault controls. Every local result retains `training_acceptance=false`; this
probe does not qualify eight-GPU communication or distributed expert ownership.

## Remaining qualification

The new [MoE semantics API](moe-routing.md) independently checks each endpoint's captured projection, top-k, dispatch and combine, retaining missing evidence or budgets. Hand-computed controls, exact rational comparisons, ties and index faults test the CPU auditor; they do not qualify GPU training.

Formal calibration still needs supported expert-internal probability multiplication and softmax arithmetic, plus independent budgets for propagation of upstream activation and weight differences. Candidate thresholds must then be fixed before evaluating untouched held-out data, and must separate healthy controls from actual training faults. Each baseline seed/case/layout ran once; paired observer noninterference does not replace independent repeatability controls. Longer sequences, more steps and the full parallel domain still require qualification.
