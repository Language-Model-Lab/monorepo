# Independent project baselines

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


This record captures backend and quant validation before consolidating their
dependencies into a root uv workspace. Test results below were supplied by the
developer during the baseline session and were not rerun when writing this file.

## Starting point

- Baseline commit: `9fe2260a3fe9c723d5a4c61ef47025d5dda8653d` (`initial commit`).
- Branch: `integration/uv-workspace`.
- Documentation prepared: October 23, 2026. Exact test execution timestamps were
  not recorded here.
- The original repositories remain preserved separately. Their histories were
  not imported into this repository.
- Each project retained its own `pyproject.toml`, `uv.lock`, and `.venv`.
- Synchronization used `uv sync --locked` for backend and
  `uv sync --locked --group dev` for quant.

## Environment

Both test environments reported Linux, CPython 3.14.4, and pytest 9.1.1. uv reports
`0.12.23 (x86_64-unknown-linux-gnu)` at documentation time, and both environments'
`pyvenv.cfg` files record uv 0.12.23 and Python 3.14.4.

| Component | Backend | Quant |
| --- | --- | --- |
| Torch, from existing lockfile | 2.14.1 | 2.14.1 |
| Transformers, from existing lockfile | 5.18.0 | 5.18.0 |
| Project environment | `backend/.venv` | `quant/.venv` |
| Test configuration | `backend/pyproject.toml` | `quant/pyproject.toml` |

The original lockfiles are the dependency-resolution references. Full installed
dependency listings exist as `backend-environment.txt` and
`quant-environment.txt` in the temporary log directory below. Their contents and
durable storage locations have not been supplied for this record.

## Test results

| Suite | Collected | Selected | Passed | Deselected | Exit code | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Backend | 29 | 29 | 29 | 0 | 0 | 6.88 s |
| Quant CPU, verified monorepo interpreter | 67 | 52 | 52 | 15 | Not separately supplied for rerun | 2.90 s |
| Quant GPU | 67 | 9 | 9 | 58 | 0 | 32.15 s |
| Quant model | 67 | 6 | 6 | 61 | 0 | 9.68 s |

All reported selected tests passed; no selected tests were reported as skipped
or errored. Deselected quant tests were excluded by marker selection, not test
failures. The initial CPU invocation returned exit code 0 but used the original
repository's interpreter; the verified rerun above supersedes it.

### Backend

Executed from `backend/`:

```bash
uv run --locked pytest -v -ra
```

The reported interpreter was `monorepo/backend/.venv/bin/python`. Coverage included
generation helpers, intervention behavior, and request/response schema validation.
This suite does not establish that the API server starts or that real-model API
inference works.

### Quant CPU

Executed from `quant/`:

```bash
uv run --locked python -c "import sys, quant; print(sys.executable); print(quant.__file__)"
uv run --locked python -m pytest -v -ra
```

Both interpreter and package locations were verified inside this checkout:

```text
monorepo/quant/.venv/bin/python
monorepo/quant/src/quant/__init__.py
```

The default pytest configuration excludes `gpu` and `model` markers. Selected
tests cover quantization, quantized linear layers, model surgery, fidelity helpers,
and benchmark utilities.

### Quant GPU

Executed from `quant/`:

```bash
uv run --locked python -m pytest -v -ra -m gpu
```

The interpreter and pytest root pointed to this checkout. Tests covered CUDA timing,
stream-read checksums, L2 flushing, rotating timing, and telemetry sampling.

### Quant model

Executed from `quant/`:

```bash
uv run --locked python -m pytest -v -ra -m model
```

The interpreter and pytest root pointed to this checkout. Tests covered exact
prompt lengths of 512/1024/4096 tokens, replacement of all 196 decoder linear
layers, preserved embedding ties, and text generation with the real model.

## Environment issues encountered

1. Backend synchronization warned that `VIRTUAL_ENV` pointed to the original
   repository's quant environment. uv ignored that environment and created the
   monorepo backend environment successfully. Deactivating the old environment
   was recommended; `--active` should not be used to target it.
2. The first quant CPU invocation used a pytest launcher reporting the original
   repository's interpreter. The copied environment was suspected to retain a
   stale launcher. Running `python -m pytest` used the monorepo interpreter and
   package successfully. Use that form for subsequent validation.
3. The GPU command included `cd quant` while the shell was already in `quant/`.
   That directory change failed, but pytest continued in the correct directory
   and all nine tests passed. Future command blocks should use `cd ... &&`
   so a directory-change failure prevents test execution.

## Evidence and remaining details

The baseline procedure directed logs into a directory created with
`mktemp -d /tmp/monorepo-baseline.XXXXXX`. The developer confirmed the directory is
`/tmp/monorepo-baseline.hFHOm2` and contains all eight files below. Whether the logs
have been copied to durable storage is not recorded here.

- `backend-environment.txt`
- `backend-tests.txt`
- `quant-environment.txt`
- `quant-cpu-tests.txt` (should contain the verified rerun)
- `quant-gpu-tests.txt`
- `quant-model-tests.txt`
- `nvidia-smi.txt`
- `nvcc-version.txt`

The supplied `nvidia-smi` output reports:

- Capture timestamp as printed: `Fri Oct 23 23:21:05 2026` (timezone unspecified).
- GPU: `NVIDIA GeForce RTX 4050 ...` (name truncated in the output).
- GPU memory: 6141 MiB total.
- NVIDIA driver and NVIDIA-SMI version: 595.91.07.
- Driver-reported CUDA version: 13.2; this does not identify the installed toolkit.

`nvcc-version.txt` reports `Command 'nvcc' not found`. Thus `nvcc` was unavailable
on the shell's PATH; the installed CUDA toolkit version is unverified. The passing
GPU tests do not establish which compiler or cached extension build was used.
Passing GPU/model tests demonstrates the reported checks succeeded on the test
machine; this is not a performance benchmark baseline.

A final clean-tree check was recommended but its output was not supplied. At
documentation time, `.gitignore` had an existing uncommitted modification; it was
left unchanged. This documentation introduces a new file. These changes should be
reviewed separately from the dependency migration. No claim is made that this
documentation describes a clean committed working tree.

## Comparing the workspace migration

After introducing the shared lockfile, repeat the four suites with their existing
test configurations and working directories. Compare selected test counts and
outcomes against this record, review dependency-version changes, and record any
differences before proceeding to packaging or model-integration changes.
