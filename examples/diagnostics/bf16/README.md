# BF16 arithmetic diagnostics

These optional examples isolate numerical differences between a Hugging Face
reference and native Slime training. They are **development diagnostics**, not
qualified acceptance references. All historical failed results and thresholds
remain unchanged. See the [measured results](../../../docs/bf16-contract.md).

The helpers implement explicit forward and first-order backward formulas with
PyTorch. No native Slime, Megatron, TE kernel or captured native output supplies
an expected training value. Only `attention_contract_probe.py --output` imports
TE to measure the actual side, after computing independent predictions.

## Scope and arithmetic

The model runners require Transformers **4.57.6**, the pinned public HF adapter,
BF16 initialization, untied Qwen3 or Qwen3-MoE weights, zero dropout, no cache,
default RoPE and no sliding window. Dependencies are checked by source hash.
They use FP32 optimizer masters, effective BF16 weights/activations, one sample
per microbatch, FP32 gradient accumulation, and division by the global number
of supervised target tokens after backward. The original HF objective,
canonical parameter mapping and optimizer remain in use.

| Module | Diagnostic change |
| --- | --- |
| `bf16_ops.py` | Explicit linear FP32 weight gradients, RMSNorm, RoPE, SwiGLU and embedding gradient writeback. |
| `contract_reference.py` | Dense ablations: `bf16_legacy`, `linear_fp32wgrad`, `fused_norm_rope_swiglu`, `math_attention`. |
| `dense_packed_reference.py` | Joint QKV and gate/up input-gradient writeback; `packed_eager`, `packed_math`, `packed_fa2`. |
| `contract_moe_reference.py` | Independent router, probability placement and expert arithmetic; progressive MoE profiles. |
| `moe_fa2_reference.py` | MoE `moe_fa2` profile with independently computed attention boundaries. |
| `attention_contract_probe.py` | Six attention ablations and a separate FP64 mathematical control. |
| `seeded_fixture.py` | Independent weight/token seeds; generates fixtures without evaluating or selecting routes. |

`packed_fa2` and `moe_fa2` support a single causal sequence per microbatch,
head dimension 32 and sequence lengths **1–128**. They model one-tile FA2 cast
boundaries, including unnormalized BF16 exponents and per-query-head GQA
gradient writeback. They do not emulate every CUDA instruction, reduction order
or exponential intrinsic. Larger tiles, other dimensions and parallel backend
arithmetic need separate qualification. CPU/GPU differences near BF16 rounding
midpoints remain possible. Gradients are analytical training VJPs; they are not
finite differences through quantization or higher-order derivatives.

The MoE router uses effective BF16 weights/inputs with strict FP32 projection,
selects its own top-k logits, and normalizes the selected logits in FP32.
Its decisions remain independent of the candidate. A changed route is retained
in the observations, not replaced with the candidate's route.

These runners temporarily replace model functions and restore them on success
or exceptions. Use a separate worker process, with no concurrent or reentrant
Qwen work inside that process. Module call counts, effective arithmetic, source
identities and restoration are recorded. Unsupported contracts fail explicitly.

## Run a new diagnostic plan

Start with a local synthetic fixture from the [Qwen3 recipe](../../README.md#synthetic-qwen3-gpu-fixtures).
Use a separate environment with `transformers==4.57.6`, and run from the
repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD/examples/diagnostics/bf16"
```

Copy the fixture spec to a new file. For dense Qwen3 replace only its reference
section with:

```json
{
  "reference": {
    "runner": "dense_packed_reference:run_reference",
    "kind": "independent",
    "device": "cuda",
    "options": {"profile": "packed_fa2"}
  }
}
```

For Qwen3-MoE use `moe_fa2_reference:run_reference` and
`{"profile":"moe_fa2"}`. Keep the original checkpoint, tokens, masks, training
settings, policy and candidate identity. Add the helper files to `artifacts`
when preparing the plan so the artifact manifest captures their content.
Use a new output directory for every changed source, backend or policy:

```bash
slime-parity plan --spec diagnostic-spec.json --output runs/bf16-diagnostic
slime-parity run --suite runs/bf16-diagnostic/suite.json
```

Improved error metrics do not qualify the reference or justify relaxing gates.
The current experiments use development inputs; calibration, held-out positive
controls and fault detection are still separate prerequisites.

## Generate new fixtures

```bash
python examples/diagnostics/bf16/seeded_fixture.py \
  --kind dense --weight-seed 1103 --token-seed 2111 \
  --template-spec runs/tiny-dense-fixture/spec.json \
  --output runs/fresh-dense-fixture
```

The default recipe has three steps, batch size 16, lengths 64/65 and explicit
masks. `--template-spec` requires the same geometry, training count and masks;
it preserves the template's policy, topology and runtime training seed while
changing checkpoint and input identities. The manifest records actual weight,
token, mask and generator hashes. Output directories are never overwritten.
Choosing seeds alone does not establish a calibration/held-out split: freeze
the experiment and selection rules before inspecting those results.

## CPU checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q examples/diagnostics/bf16/tests
```

The 84 checks cover independent FP64 formulas, joint-projection mappings and
VJPs, first-order attention boundaries, three-step optimizer trajectories,
restoration after failures, unsupported shapes and generation without model
execution. The dedicated CI job pins Transformers 4.57.6. These checks do not
establish GPU or parallel numerical acceptance.

## Replay AdamW from captured state

The report-only NumPy tools read sealed canonical captures and independently
replay each side from its own gradients and previous state. They accept any
complete parameter catalog, with FP32 captured optimizer tensors. The current
formula contract requires AdamW LR `1e-5`, betas `[0.9,0.999]`, epsilon `1e-8`,
zero weight decay/clipping, and fresh consecutive steps. The base audit needs
at least two steps; the scalar-rounding variant covers only the first three.

```bash
python examples/diagnostics/bf16/adamw_replay_audit.py \
  --suite runs/bf16-diagnostic/suite.json --output runs/adamw-replay.json
python examples/diagnostics/bf16/adamw_hyperparam_replay.py \
  --suite runs/bf16-diagnostic/suite.json \
  --baseline-audit runs/adamw-replay.json --output runs/adamw-scalars.json
```

Both tools verify plan, process and rank receipts, raw hashes and canonical
ownership. Runtime source/checkpoint artifact files are not rehashed by default
when replaying an archived capture on another machine. Use the base audit's
`--verify-live-artifacts` on the original environment to verify those files too.
Every output must be new. Missing/invalid evidence and exact update/continuity
violations fail; `complete_report_only` supplies no numerical acceptance gate.
The scalar variant checks that it is reading the same sealed evidence as the
base audit. Formula variants intentionally retain their stated operation order,
without claiming to emulate CUDA FMA or approximate reciprocal instructions.

The main CPU suite includes 23 replay checks covering real PyTorch trajectories,
Decimal scalar controls, canonical sharding, checksum failures, state
continuity and injected optimizer faults. Fault tests validate the audit
mechanism; they are not held-out model acceptance results.
