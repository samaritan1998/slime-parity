# GPU validation, 2026-10-26

[English](gpu-validation.md) | [简体中文](gpu-validation.zh-CN.md)

Native Slime completed three optimizer steps for synthetic Qwen3 dense and MoE
models on one and eight GPUs. The independent HF comparisons returned
**`numerical_failure`** under the original diagnostic policy. Execution evidence
does not qualify these models as numerically correct.

The [machine-readable summary](gpu-validation-2026-10-26.json) contains the
configuration rows, gate counts and per-step logits diagnostics. It is an
allowlisted summary, without raw tensors, logs or runtime paths.

## Tested environment and contract

| Component | Observed version or setting |
| --- | --- |
| Slime | [`18868a79b0b083a093798a94ed0f8cb4f71b9c14`](https://github.com/THUDM/slime/tree/18868a79b0b083a093798a94ed0f8cb4f71b9c14) |
| Megatron | [`1dcf0dafa884ad52ffb243625717a3471643e087`](https://github.com/NVIDIA/Megatron-LM/commit/1dcf0dafa884ad52ffb243625717a3471643e087), core `0.16.0rc0`, plus the public patch below |
| Python / PyTorch / CUDA | `3.12.3` / `2.11.0+cu129` / `12.9` |
| Transformer Engine | `2.10.0` |
| Transformers / HF Hub / tokenizers | `4.57.6` / `0.36.0` / `0.22.2` |
| FlashAttention / FlashAttention 3 | `2.7.4.post1` / `3.0.0b1` |
| GPU domain | One node, eight devices, CUDA compute capability `10.0` |
| Attention | Flash; `NVTE_FLASH_ATTN=1`, `NVTE_FUSED_ATTN=0`, `NVTE_UNFUSED_ATTN=0` |
| Precision | Slime BF16; HF eager FP32; identical BF16-rounded initial values |

Megatron includes Slime's public
[`v0.5.12.post1/megatron.patch` at `bf14dc21`](https://github.com/THUDM/slime/blob/bf14dc21f9500746447f2572d0692e981c4d2a7e/docker/patch/v0.5.12.post1/megatron.patch).
Its SHA-256 is `976fd8bbe1f4e386be97e35353efdbfb592dacaea9797c5a81ea324d3c2c5b77`.
All 865 Python files in the runtime source matched the public upstream plus
that patch; using the upstream commit alone does not reproduce this environment.

The runtime had a cuDNN version-probe mismatch: PyTorch expected `9.17.1`, while
the installed cuDNN package was `9.16.0.29`. The probe error was retained in
private runtime evidence and execution used Flash attention. This run does not
establish cuDNN attention compatibility. Optional HF kernel discovery was
disabled in the isolated Transformers environment; no remote model code was used.

Both models use seed 417, vocabulary 512, hidden size 128, four layers, four
attention heads, two KV heads and no dropout. The dense FFN size is 256. MoE
uses four experts, top-2 routing, expert FFN size 128 and zero auxiliary/z-loss.
The independent maps contain 39 dense or 79 MoE canonical trainable parameters.

Each case uses 16 samples and schedules three AdamW updates; completed
configurations performed all three updates. AdamW uses LR `1e-5`, betas
`[0.9, 0.999]`, epsilon `1e-8`, no weight decay or clipping. The objective is
cross-entropy normalized by global supervised tokens. The two cases have
sequence lengths 64 and 65; the latter also exercises loss-mask boundaries.

## Completed smoke and mixed runs

| Model | Configuration | Case | Execution | Failed tensor/step gates |
| --- | --- | --- | --- | ---: |
| Dense | One GPU, all degrees 1, SP off | text | Complete | 71 / 708 |
| Dense | Eight GPUs, TP2 CP2 PP2, SP on, EP1 ETP1 | text | Complete | 82 / 708 |
| Dense | Same mixed configuration | masked-boundaries | Complete | 89 / 708 |
| MoE | One GPU, all degrees 1, SP off | text | Complete | 526 / 1428 |
| MoE | Eight GPUs, TP2 CP2 EP2 PP2, SP on, ETP1 | text | Complete | 517 / 1428 |
| MoE | Same mixed configuration | masked-boundaries | Complete | 532 / 1428 |

Every listed run returned `numerical_failure`. Full forward outputs, loss,
reduced gradients, initial/resulting master parameters, actual updates and Adam
moments were captured for all three steps. Rank receipts and raw checksums were
validated, including the native process-group memberships.

The full matrix requests every 1/2 combination of TP, CP, EP, PP and ETP, both SP
values, world size 8 and VPP=1 for both cases. Dense has 48 eligible rows, 16
invalid rows and 64 inapplicable rows. All 48 eligible dense rows completed with
valid evidence and returned `numerical_failure`: 3,874 of 33,984 tensor/step gates
failed (494 gradient, 2,644 update and 736 first-moment gates). All forward, loss,
initial/resulting parameter and second-moment gates passed their declared limits.
The largest logits error was `0.014046`, and the largest loss error was `0.00017839`.
MoE has 80 required rows and 48 invalid rows. All required rows were attempted:
48 completed with valid evidence and numerical failures; 32 stopped during
execution and remain unverified. The latter comprise 16 backend rejections
requiring SP, eight incomplete initial optimizer-master captures and eight
expert gradient replica disagreements. Among the completed rows, 23,962 of
68,544 tensor/step gates failed. The largest logits error was `0.100845`, and
the largest loss error was `0.000278148`. All loss and initial-parameter gates
passed; the original failed results remain unchanged.

| Full matrix | Requested rows | Required rows | Completed | Execution failures | Numerical passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 128 | 48 | 48 | 0 | 0 |
| MoE | 128 | 80 | 48 | 32 | 0 |
| Total | 256 | 128 | 96 | 32 | 0 |

Every completed required row is `numerical_failure`; every execution failure
remains `unverified`. There are no pending required rows. Invalid and
inapplicable rows remain visible in the structured summary.

## What the discrepancies show

The pinned `MoELayer.forward` rejects training whenever attention TP is greater
than one and SP is disabled, regardless of EP or ETP. The matrix planner's
generic constraints did not exclude every such row; `eligible` means those
declared constraints passed, with native preflight still required. Backend
rejections remain required rows with `execution_failure` / `unverified`; they
are neither numerical failures nor retroactively removed from coverage.

With the pinned backend, MoE configurations with TP=1, EP=1 and ETP=2 failed
initial optimizer-master coverage before training. Source inspection found
that `TEGroupedLinear` shards expert weights using expert TP, but classifies
them for the ordinary DP optimizer whenever EP=1. With different attention and
expert TP degrees, that optimizer group can mix different expert weight slices.

A CPU reproduction using the pinned checkpoint factory and optimizer range
algorithms reproduced the missing half of each FC1 gate/up parameter for PP=1
and PP=2. Raw model parameters had complete coverage. Assigning the synthetic
fixture to expert DP restored master coverage; TP=ETP=2 controls were already
complete. This identifies a grouping mismatch in the pinned code; the CPU
fixture does not substitute for a dump of the failed GPU's actual buffer ranges.
The GPU runs retain the original backend, training semantics and coverage
checks. These rows remain execution failures with unverified numerical results.

The reverse configuration, TP=2/EP=1/ETP=1 with SP enabled, reached complete
initial coverage but failed the expert gradient replica-agreement check before
the first update. This is consistent with the grouping concern, but a checksum
disagreement alone gives neither the difference magnitude nor a complete
runtime root-cause proof. It remains an evidence failure, not a measured
numerical gate result.

In the dense one-GPU text case at step 0, update differences reached about
`2e-5` with LR `1e-5`.
All 2,216 elements above `1.8e-5` had opposite gradient signs. The observed
updates and moments were consistent with each side's own gradients, and
`update = parameter - initial` held exactly. This explains how small gradient
differences can produce large update differences; it does not validate the
underlying gradients or change the failed gates.

MoE also differs before the first optimizer update: the one-GPU logits maximum
absolute difference from HF was `0.100845`, at `[sample=14, token=33, vocab=338]`.
The mixed configuration's difference at that position was `0.098892`.
Comparing native one-GPU and mixed runs directly gave maximum logits difference
`0.065430`; their router-gradient relative L2 was `6.343%`, and one expert tensor
reached `12.045%`. These direct comparisons are diagnostics, not an independent
reference or an alternative acceptance result.

Separate route probes retained the original three-step training. Each of six
reference/native probe stages matched its original run bit for bit on all
476 first-step canonical tensors: 2,856 stage/tensor comparisons in total.
Each side covered all four layers and all 16 samples: 4,096 layer/sample/token
positions (`4 × 16 × 64`) for text or 4,160 (`4 × 16 × 65`) for masked-boundaries.
The mixed masked case discarded 192 padding observations. Every layer selected
all four experts. HF selections were derived using its routing formula; native
selections and weights were captured from the actual router return.

| First-step case | Changed expert sets by layer, HF → native |
| --- | --- |
| Single-GPU text | 4 / 5 / 3 / 11 |
| Mixed text | 5 / 6 / 5 / 5 |
| Mixed masked-boundaries | 2 / 6 / 7 / 5 |

Native single-GPU versus mixed text changed sets at 1 / 1 / 4 / 8 tokens by
layer. CPU FP32 projections of the captured inputs using the identical initial
router weights reproduced every observed top-2 set, with maximum gate-logit
residual `1.788e-7`. The differences are already present in the router inputs;
this does not identify which upstream operation caused them.

A separate CPU forward-only experiment forced the native expert sets while
recomputing selected weights from the current HF logits. It used PyTorch
`2.14.1+cpu` and Transformers `4.57.6`. Untouched CPU HF differed from GPU HF by
at most `7.153e-7`; copied-forward, forced-HF-set and restoration controls were
bitwise identical to the untouched CPU baseline.

| First-step case | Max logits difference from native, before → after intervention | Tokens above 0.025, before → after |
| --- | --- | --- |
| Single-GPU text | 0.100845 → 0.007553 | 20 → 0 |
| Mixed text | 0.098892 → 0.008276 | 18 → 0 |
| Mixed masked-boundaries | 0.056067 → 0.008142 | 19 → 0 |

This table covers step 0, before the first update. The mixed masked case's
three-step maximum difference, `0.081124`, occurred at step 1; this intervention
did not cover later steps.

Changing the expert sets accounts for much of the observed forward discrepancy
in these cases. Residual relative L2 remains about 0.64–0.69%. The intervention
changes routing and performs no backward or optimizer steps; it cannot qualify
the original runs, validate gradients/updates or establish an upstream GPU fault.

The step-0 aggregate comparison between the one-GPU and mixed text
configurations showed no factor-of-two CP/EP gradient scaling signature.
That check cannot rule out local mapping, routing or numerical errors.
The original [policy](../examples/policies/tiny-qwen3-diagnostic.json) remains
unchanged. It is a strict diagnostic policy, not a calibrated BF16 recommendation.

## Fixes and reproduction

The structured summary includes the frozen library source hashes for each run.
All 26 library Python files in the full MoE matrix match this publication.
The dense matrix used an earlier snapshot without the additional compute-mode
preflight and environment fields described below; its capture and comparison
code is identical. The completed CPU regression covers the published source.

The GPU work found a THD capture mismatch: Slime slices a sample for CP before
padding the local stream for TP. Capture now follows that order and excludes
padding sequences from global logits. The change was independently checked
against the pinned Slime batch builder in 810 CP-rank-local layout cases across
216 configurations on CPU; these are layout checks, not additional GPU runs.

Runtime metadata now records an unavailable cuDNN probe without aborting
unrelated GPU execution. The comparator also reuses validated file handles
within a tensor; fragmented TP layouts no longer reopen files for every interval.
Path restrictions, truncation checks and stream cleanup remain covered.

The native preflight also checks known compute modes that parameter dtype cannot
detect: FP8/FP4 compute or parameter storage, Kitchen recipes, quantized saved
activations, active TE module overrides, fake INT4, forced routing and requested
routing replay. These require a dedicated
reference/adapter. It records relevant environment switches and rejects an
unsupported mode instead of silently changing it.

The CPU regression suite passed 124 tests. The synthetic fixture generator
reproduced every GPU fixture weight exactly, along with the same inputs, mapping
and training arguments. See [the recipe](../examples/README.md#synthetic-qwen3-gpu-fixtures).
The generator defaults to no policy; the diagnostic policy must be selected
explicitly. New model architectures, larger degrees, VPP, multi-node execution,
RL and multimodal objectives remain outside these results.
