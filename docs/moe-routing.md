# MoE routing semantics

[English](moe-routing.md) | [简体中文](moe-routing.zh-CN.md)

`slime_parity.diagnostics.moe_routing` independently checks each endpoint's router projection, top-k selection, expert dispatch and output combination from its own captured operands. It needs only NumPy. It does not run training or select an acceptance tolerance.

**A local `pass` is never training acceptance.** Every report retains `training_acceptance=false`. Missing observations, arithmetic contracts or independent budgets produce `inconclusive`; a known failure remains visible alongside missing evidence. Malformed shapes, dtypes, identities, indices or nonfinite values raise `EvidenceError`.

## Python API

This runnable, hand-computed example demonstrates the interface; it is not a GPU calibration case:

```python
import numpy as np
from slime_parity.diagnostics import moe_routing as mr

inputs = np.array([[2, 2, 0]], dtype=np.float32)
endpoint = mr.EndpointEvidence(
    label="synthetic",
    coordinates=((0, 1, 0, 0),),  # step, layer, sample, token
    expert_ids=(0, 1, 2),
    inputs=inputs,
    weights=np.eye(3, dtype=np.float32),
    logits=inputs.copy(),
    routing_map=np.array([[True, True, False]]),
    probabilities=np.array([[0.5, 0.5, 0]], dtype=np.float32),
    contract=mr.RoutingContract(
        top_k=2,
        input_dtype="bfloat16",
        weight_dtype="bfloat16",
        logit_dtype="float32",
        projection_arithmetic="fp32_mul_add_or_fma",
        selection_domain="logits",
        probability_rule="selected_softmax",
    ),
)
result = mr.audit_endpoint(endpoint)
assert result["checks"]["projection"]["status"] == "pass"
assert result["checks"]["topk"]["status"] == "pass"
assert result["status"] == "inconclusive"  # No probability budget declared.
```

BF16 operands use exact FP32 promotions. Nonzero low 16 bits are rejected; relabeling ordinary FP32 data as BF16 is invalid evidence.

| Interface | Scope |
| --- | --- |
| `audit_endpoint` | Own-operand independent projection, legal top-k on actual selection scores, probability support and normalization. |
| `audit_pair` | Aligned identities, a declared cross-end budget and independently computed stable reference sets. Candidate routes never become reference routes. |
| `routing_from_slots` | Sparse matrices from actually captured expert slots/probabilities; duplicate or out-of-range slots fail. |
| `audit_dispatch` | Complete token/expert pair permutation, expert order/counts, input/probability transport and actual inverse scatter indices. |
| `audit_combine` | Actual expert outputs replayed under explicit multiplication, accumulation and output boundaries. |
| `audit_layer` | A fixed routing/dispatch/combine catalog; missing observations cannot disappear from its overall status. |

Token coordinates must be unique. `expert_ids` identifies the canonical expert columns; paired endpoints must agree. `DispatchEvidence.row_expert` indexes those columns. Partial EP shards cannot stand in for the complete expert domain. Zero-hit experts retain their identity and zero count without fabricated rows or outputs.

`audit_pair` aggregates projection and route-stability checks only. It does not replace the probability, dispatch and combine catalog in `audit_layer`.

## Numerical premises

Projection supports bias-free dots with effective BF16 operands and ordinary round-to-nearest FP32 multiplication/addition or FMA. It allows at most `2H` roundings for width `H`, using `u32=2^-24` and a conservative `gamma_(2H)` bound. The independent FP64 sum is not exact real arithmetic: its separate summation error and the error in the absolute-product sum are included, with upward-rounded bounds.

Nonzero subnormal operands, possible subnormal intermediate sums, overflow and truncated accumulators are outside the supported domain. Requiring every BF16 product to be an integer multiple of `2^-126` conservatively rules out subnormal cancellation under arbitrary ordinary accumulation orders. Exclusion is inconclusive, not a kernel failure or permission to enlarge the budget. Kernel compliance remains a caller-supported premise; GPU instructions are not observed by this API.

Products are blocked over token/expert dimensions. A temporary product block contains at most `max(2**20, H)` FP64 elements; inputs, outputs and other intermediates still consume memory. This is not a production-scale performance qualification.

A uniquely stable top-k set requires every selected lower interval endpoint to exceed every excluded upper endpoint. Exact cutoff ties permit multiple legal sets, without imposing an expert-index tie-break. This same-operand bound covers neither upstream activation drift nor trained-weight differences.

`DeclaredBound(absolute, source)` stores an externally frozen absolute budget and its derivation identity; it cannot certify that derivation. A cross-end budget must cover upstream differences and candidate projection error relative to the independently recomputed reference projection. No budget is fitted to observed residuals. Without it, even identical route sets remain inconclusive.

Dot-product gamma bounds do not cover softmax exp/division. Probabilities, probability sums and optional full-softmax selection scores need separate budgets. FP64 softmax is an independent numerical reference, not an exact-real oracle. If selection uses rounded full-softmax scores, those actual scores must be captured; logits cannot silently replace them.

## Capture integration

This API does not load raw archives or install training hooks. Before constructing evidence, verify source/file identities, required step/layer/sample/token coverage, expert mappings and paired noninterference against uninstrumented training. Checksums and matching expert names alone do not prove semantic correctness.

Dispatch rows and inverse indices must come from actual calls. Captured permutations and chunk-sort arguments may reconstruct row identities, retaining the derivation. Generating a desired permutation from the router map is not evidence that the runtime used it. Input/probability transport compares exact bytes, including signed zero.

The dispatch contract covers a complete logical `T*K` domain, without capacity dropping or padding. It does not verify EP/TP shards, SP/CP ownership, replicas, all-to-all traffic or distributed reductions.

When probabilities are applied before expert FC2, captured expert outputs are already weighted. The combine check can then verify inverse transport and summation, while internal probability multiplication remains inconclusive. It does not manufacture unweighted outputs by dividing by probabilities.

The explicit BF16 rounding helper uses FP32 → BF16. A combine contract using it must support that intermediate boundary; it is not an arbitrary direct FP64 → BF16 rounding oracle.

## Validation scope

The [fresh-seed W1 observations](cross-seed-diagnostics.md#actual-w1-moe-boundaries)
exercise this API on captured GPU boundaries with complete paired noninterference
checks. The [optional observer](../examples/diagnostics/bf16/moe-w1-capture.md)
supports the documented tiny fixture, not arbitrary models or distributed ownership.

```bash
python -m pytest -q tests/test_moe_routing.py
```

Tests include hand-computed controls, exact rational dot products, BF16 rounding boundaries, ties, wrong weights/slots, duplicated or omitted token/expert pairs, wrong inverse indices, combine faults and zero-hit experts. These faults test the auditor, not end-to-end GPU training fault separation.

Local routing checks do not replace forward/backward, gradient, optimizer-state or full parallel-matrix acceptance. The frozen per-tensor policy and [compatibility boundaries](compatibility.md) still apply.
