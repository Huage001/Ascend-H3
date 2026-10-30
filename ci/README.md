# Optional CPU CI

`github-actions.example.yml` is a template, not an active workflow. To enable
it with an account authorized to manage workflows, copy it to
`.github/workflows/cpu-checks.yml` and commit that change.

Run its checks locally with Python 3.12:

```bash
python -m pip install -e . ruff==0.17.3
python -m unittest discover -s tests -q
ruff check src scripts tests
```

These are CPU contract checks. They do not build CANN kernels, load H3 weights,
qualify NPU inference or reproduce the recorded speedups.
