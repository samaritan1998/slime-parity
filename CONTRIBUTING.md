# Contributing

Use an issue to describe a reproducible failure or a proposed adapter. Include
package versions, the requested topology and a small synthetic input whenever
possible. Reports can contain local paths, model data and runtime metadata;
review them before attaching them to a public issue.

## Development

```bash
python -m pip install -e '.[dev,hf]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

For numerical changes, add a regression that fails under the old implementation
and checks the measured result or acceptance decision. For native adapters,
preserve independent parameter catalogs, frozen inputs, actual process-group
evidence and complete tensor coverage. Unsupported execution remains unverified.

CPU tests do not establish Slime GPU compatibility. A GPU compatibility result
needs pinned model and software versions, the full declared matrix and matching
reference semantics. Share a reproducible public fixture instead of model weights
or raw reports containing sensitive data.

Keep changes focused and describe the problem, resulting behavior and validation
in the pull request. Contributions are licensed under the repository's
Apache-2.0 license.
