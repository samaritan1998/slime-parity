# Experimental Megatron expert-DP patch

`megatron-expert-dp-classification.patch` corrects `TEGroupedLinear` parameter
grouping when EP=1 and attention TP differs from expert TP. Expert parameters then
use expert-DP gradient buffers and distributed optimizer ownership. The patch
does not change the `expert_parallel` communication flag or the backend SP guard.
The library does not apply this patch automatically.

The reviewed base is Megatron
[`1dcf0dafa884ad52ffb243625717a3471643e087`](https://github.com/NVIDIA/Megatron-LM/commit/1dcf0dafa884ad52ffb243625717a3471643e087)
with Slime's public
[`v0.5.12.post1/megatron.patch` at `bf14dc21`](https://github.com/THUDM/slime/blob/bf14dc21f9500746447f2572d0692e981c4d2a7e/docker/patch/v0.5.12.post1/megatron.patch)
already applied. The prerequisite patch has SHA-256
`976fd8bbe1f4e386be97e35353efdbfb592dacaea9797c5a81ea324d3c2c5b77`.

Use a separate backend checkout prepared from that base. From the slime-parity
repository root:

```bash
git -C /path/to/megatron-experiment apply --check \
  "$PWD/examples/patches/megatron-expert-dp-classification.patch"
git -C /path/to/megatron-experiment apply \
  "$PWD/examples/patches/megatron-expert-dp-classification.patch"
```

The patch changes one file, `megatron/core/extensions/transformer_engine.py`.
Its SHA-256 changes from
`ea756922b545e144c4bcc4c9a8260f254bb80fad32411f0eb4dc1241eb40bca1`
to `d1505d230692fa9fb85be60582fb66aa7742a81c1ddb5bc799e2f81a459e4800`.
The patch file SHA-256 is
`b4e6151383fd3ee3787e93c4ff54bfb0c92d9009cf9da0e9a0e78a616a2cc712`.
Other versions require a fresh review.

Select this checkout through `candidate.options.megatron_source` in a new spec
and create a new plan. Retain the original backend's plans and failed evidence.
The scoped GPU comparisons and remaining numerical failures are recorded in
[the baseline controls](../../docs/baseline-controls.md). A completed run does
not establish numerical acceptance or full-matrix compatibility.
