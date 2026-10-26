# Baseline and backend controls, 2026-10-26

[English](baseline-controls.md) | [简体中文](baseline-controls.zh-CN.md)

These follow-up experiments add explicit HF precision controls, locate an early
dense-model discrepancy and test an expert-parameter ownership patch. **Numerical
acceptance is still not achieved.** The original full-matrix results remain
unchanged. The [structured summary](baseline-controls-2026-10-26.json) contains
allowlisted metrics and public source identities, without runtime paths or logs.
The tests use one, four or eight GPUs on one node, with the pinned environment
and synthetic fixtures from the [original record](gpu-validation.md).

## Independent reference controls

Three public HF precision controls completed three optimizer steps on the same
single-GPU dense Qwen3 text fixture: 16 samples of length 64. **All three remain
`numerical_failure` under the original policy.** The structured summary records
the per-step metrics, unchanged policy hash and all 26 library source hashes for
commit `4d4e12422d0f0cdd1fbbeb9cbae9420defb92331`.

`fp32` uses FP32 parameters and compute, a full 16-sample batch and normalization
before backward. `autocast` changes compute to BF16 while retaining FP32
parameters. `bf16_micro_after` uses BF16 parameters, microbatch size 1 and global
token normalization after FP32 gradient accumulation. All retain FP32 optimizer
masters and the constructed buffer precision. BF16 autograd gradients can round
before accumulation; this does not reproduce TE fused weight-gradient arithmetic.

| Control | Failed gates, total (gradient / update / first moment) | Max logits error | Max gradient relative L2 | Max update error |
| --- | ---: | ---: | ---: | ---: |
| `fp32` | 71 / 708 (5 / 57 / 9) | 0.0130851 | 0.0102591 | 1.99946e-5 |
| `autocast` | 82 / 708 (12 / 55 / 15) | 0.0117188 | 0.0107712 | 1.99955e-5 |
| `bf16_micro_after` | 76 / 708 (8 / 57 / 11) | 0.0112305 | 0.0103437 | 1.99974e-5 |

Maxima span steps 0–2. Gradient relative L2 concatenates all parameter gradients
within a step; acceptance still checks each tensor separately. Forward, loss,
initial/resulting parameter and second-moment gates passed their declared limits.

Separate first-step probes covered 75 sites × 16 samples × 64 tokens per side
and control. Each of six probe stages matched its preceding uninstrumented
control on all 708 training tensor/step observations. Public FP32 and autocast
runs also matched those controls byte for byte. For BF16, the public reference
matched 706 of 708 observations: only losses at steps 0 and 1 differed, each by
`4.76837158203125e-7` (one FP32 ULP). Its native stage matched all 708. The layer
probes therefore retain their original control provenance. The earlier helper
divides the reported loss by a Python integer; the public runner uses a device
FP32 denominator, consistent with this scalar rounding difference. Logits,
gradients, master parameters, updates and optimizer moments were unchanged.

FP32 and autocast first differed at `layer0.input_norm`, whose HF output was
FP32 and native output BF16. With BF16 parameters, embedding, input norm, Q/K/V
projections and Q/K normalization were equal; the first difference was RoPE
(`layer0.q_rope` and `layer0.k_rope`, maximum absolute difference `0.015625`).
An independent CPU formula with BF16 coefficients and intermediate rounding
reproduced all **786,432 HF Q/K RoPE output elements across four layers** from
their captured inputs. FP32 intermediate arithmetic closely approximated native
outputs, with residual differences. This supports a rounding diagnosis, but does
not prove GPU kernel or backward correctness or alter the failed results.

The separate GPU RoPE arithmetic diagnostic also completed three steps. It
returned **80/708 failed gates**, compared with 76/708 for ordinary
`bf16_micro_after`. Its maximum logits difference was `0.009765625` at each step.
Gradient relative L2 changed from `0.0103437 / 0.0102481 / 0.0101069` to
`0.0101181 / 0.0100270 / 0.0100895`; update relative L2 changed from
`0.121889 / 0.0336795 / 0.0199673` to `0.120477 / 0.0323532 / 0.0189817`.
Some aggregate errors decreased while more individual gates failed. This is
not an acceptance fix. The native stage remained byte-identical on all 708
observations. The reference recorded 48 coefficient calls, 192 rotation calls
and successful attribute restoration.

The [reproducible RoPE diagnostic](../examples/diagnostics/README.md) preserves
the tested executable structure; publication changes only its module docstring
and one error message. It remains an explicit, altered-reference experiment.
Six CPU tests passed with Transformers 4.57.6. Four integration tests explicitly
skip on other HF versions; the two pure-PyTorch rotation tests still run.
The HF-control implementation's CPU regression passed 150 tests, including 36
HF tests, at commit `4d4e124`. With the diagnostic examples included, the full
CPU suite passed 156 tests under Transformers 4.57.6. Under Transformers 5.13.0,
the two RoPE operator tests passed and four version-specific integration tests
were explicitly skipped.

## Expert parameter ownership

Four original/patched pairs used the same tiny MoE text fixture, 16 samples of
length 64, three planned AdamW steps, the original policy and the library snapshot at
`a57421d`. CP=EP=VPP=1 throughout. The runtime versions, weights and optimizer
contract match the [original GPU record](gpu-validation.md). The full matrix
was not rerun with this patch; its original 128 required rows and failures
remain unchanged.

| GPUs | TP | ETP | SP | PP | Original | Patched |
| ---: | ---: | ---: | --- | ---: | --- | --- |
| 4 | 1 | 2 | off | 1 | Initial masters missing 393,216 elements | Complete; 468/1,428 gates fail |
| 4 | 2 | 1 | on | 1 | Expert gradient replicas conflict on 786,432 elements | Complete; 507/1,428 gates fail |
| 4 | 2 | 2 | on | 1 | Complete; 531/1,428 gates fail | All 1,428 observations byte-identical; same failures |
| 8 | 1 | 2 | off | 2 | Initial masters missing 393,216 elements | Complete; 468/1,428 gates fail |

The probe read actual DDP buffers, optimizer groups and owned master/gradient
ranges before the ordinary checks. For TP1/ETP2, the original implementation
put expert shards into dense-DP groups containing different weight slices;
the patch selected expert-DP groups. For TP2/ETP1, original initial coverage
was complete, but all 786,432 expert gradient elements disagreed between
canonical replicas. Selecting expert DP removed the conflicts in this test.

Each patched run has sealed evidence for all 1,428 tensor/step observations,
237 checks of `update = parameter - initial`, and 158 checks of master continuity
between steps. All 128 rank-local expert parameter records use the expected
buffer group. These are ownership and execution checks, **not numerical passes**.
The failed original runs have hash-checked diagnostic probes but no completed
stage seal. The probe does not independently verify every collective operation
or its scaling.

The [experimental patch and application instructions](../examples/patches/README.md)
pin the exact backend source. It changes one classification in `TEGroupedLinear`;
the library does not apply it automatically. It preserves communication flags
and the backend's SP requirement. The separate CPU classification/geometry
checks cover 18 and 20 cases, respectively. Removing the SP guard needs a
separate token-ownership and gradient-normalization implementation; this patch
does not make those rejected configurations executable.

## Remaining acceptance work

1. Establish the intended BF16 arithmetic contract and validate forward and
   backward operators independently. RoPE localization alone leaves the dense
   baseline failing; HF BF16 gradient storage also differs from TE fused wgrad.
2. Establish tolerances using verified controls, held-out seeds/inputs and
   negative fault-injection tests. Freeze a new policy before acceptance; keep
   the failed diagnostic policy and results unchanged.
3. Rerun the entire declared TP/CP/EP/SP/PP/ETP matrix, including masked inputs
   and mixed configurations, with the qualified reference and backend.
   Unsupported configurations remain visible and unverified.
