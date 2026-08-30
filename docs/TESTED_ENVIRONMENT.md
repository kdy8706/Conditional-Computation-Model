# Tested environment

The public checkpoint-evaluation route was tested on Windows PowerShell with
CPython 3.12, CPU PyTorch, and the dependency ranges declared in
`pyproject.toml`.

```text
Python 3.12
pytest: 13 passed, 1 optional local-artifact regression test skipped
```

The optional skipped test requires user-supplied historical artifact paths and
is not part of the documented Release route. CPU execution is supported; CUDA
is optional. The 1,000-epoch retraining configuration is a research run rather
than a quick demonstration, so run time depends materially on hardware.

For a short installation check, use `python --version`, install the editable
package, run `python -m pytest -q`, then use
`python scripts/preflight_release.py`.
