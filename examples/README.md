# Examples

`cpu-suite.json` is a complete runnable fixture: two input cases, three optimizer
steps and one/two-process candidates. Run from an installed environment:

```bash
slime-parity validate --spec examples/cpu-suite.json --output runs/cpu
```

`make_slime_suite.py` generates the executable suite schema from local HF weights,
an independently reviewed parameter map and a model's existing Slime argv. It
enumerates TP/CP/EP/SP/PP/ETP/VPP, retaining invalid/inapplicable rows. It does not
invent model arguments or numerical tolerances. See `--help` and the
[integration guide](../docs/model-integration.zh-CN.md).

Neither CPU fixtures nor generated plans establish Slime GPU compatibility.
