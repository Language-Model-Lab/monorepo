# KernelBackend

An AI language model experimentation project combining a FastAPI backend with
weight only quantization and CUDA benchmarking for Qwen3-0.6B.

The repository uses a uv workspace so each component declares its own dependencies,
while one root `uv.lock` and root `.venv` provide a shared development environment.
The backend currently loads the standard model through Transformers and connecting
it to quant's quantized inference path is a also an integration step.

## Repository layout

| Path | Role |
|---|---|
| `backend/` | API, generation settings, model loading, and attention/MLP interventions |
| `quant/` | Quantization library, CUDA sources, fidelity checks, and benchmark scripts |
| `notebooks/` | Exploratory model and intervention notebooks |
| `documentation/` | Progress reports, meeting notes, mockups, abstracts, and shared project documentation |
| `pyproject.toml` | Workspace membership |
| `uv.lock` | Shared, reproducible dependency resolution; commit changes to this file |

## Set up the shared environment

First install uv and Python 3.14, run the following from the repository root:

```bash
uv sync --locked --all-packages --all-groups
```

- This installs both components and their development dependencies into the root
`.venv`. `--locked` rejects an old lockfile instead of changing it.
- Make sure to run this after cloning or pulling dependency changes.

- Manual environment activation is unnecessary. If an environment from another
checkout is active, you can `deactivate` it first. Use `uv run` to put shared environment
tools, including Ninja for CUDA extension builds, on PATH.

- Verify the interpreter and quant package location:

```bash
uv run --locked --all-packages --all-groups python -c \
  "import sys, quant; print(sys.executable); print(quant.__file__)"
```

Expect the root `.venv/bin/python` and this checkout's
`quant/src/quant/__init__.py`.

## Start the API

Run from the repository root:

```bash
uv run --locked --all-packages --all-groups python -m uvicorn kernel_backend.api:app
```

- The server exposes `POST /generate`; interactive API documentation is available at
<http://127.0.0.1:8000/docs>. Example requests are in `backend/JSON_examples/`.
Startup loads `Qwen/Qwen3-0.6B` and may download its tokenizer and model weights.
Allow sufficient memory for the model. Device placement is selected automatically.

- Backend is installed as the `kernel_backend` Python package, so this entry point
also works from `backend/`. Importing the API still loads the model immediately.
The response latency fields currently contain placeholder values and should not
be used as benchmark measurements.

## Run tests

- Run each block from the repository root. Subshells keep your shell in the root
after each command. Each suite retains its component's working directory and
pytest configuration, while `uv run` selects the shared environment.

### Backend

```bash
uv run --locked --all-packages --all-groups python -m pytest -c backend/pyproject.toml backend/tests -v -ra
```

From `backend/`, use `uv run --locked --all-packages --all-groups python -m pytest -v -ra`.

- Validates schemas, generation helpers, and interventions. The migration baseline
is **29 passed**. This suite does not validate server startup or real-model API
inference.

### Quant CPU

```bash
(
  cd quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra
)
```

- Validates quantization, layer replacement, fidelity helpers, and benchmark
utilities. Tests use small randomly initialized models where needed. The default
configuration excludes GPU and real-model tests: **52 passed, 15 deselected** at
the migration baseline.

### Quant GPU

```bash
(
  cd quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra -m gpu
)
```

- Validates CUDA timing, streaming reads, and telemetry. It requires a CUDA capable
NVIDIA GPU.
- Extension builds also require a compatible CUDA toolkit/compiler and
host C++ compiler. 
- Ninja is included in quant's dependencies
- CUDA extension sources currently target compute capability 8.9 by default. 
- First use may compile an extension. The migration baseline is **9 passed, 58 deselected**; check for skipped tests when validating a machine without CUDA.

### Quant real model

```bash
(
  cd quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra -m model
)
```

- Validates prompt construction, quantized model surgery, preserved embedding ties,
and generation using the real Qwen3-0.6B checkpoint. 
- Requires a CUDA GPU and model weights, which may be downloaded. 
- The migration baseline is **6 passed, 61 deselected**.

- These counts describe the migration checkpoint; update them as tests are added.
- Record in [independent baselines] and [workspace validation] dedicated documentation

## Quant experiments

- See [quant's README](quant/README.md) for fidelity and performance experiment
commands. Run scripts from `quant/` through the shared environment, for example:

```bash
(
  cd quant &&
  uv run --locked --all-packages --all-groups python scripts/quant_error_report.py
)
```

*Note:* Preserve results and validation logs separately when they are needed for comparison.
Performance experiments require controlled hardware conditions which means passing tests
alone does not establish benchmark equivalence.

## Dependency changes

Edit the relevant component's `pyproject.toml`, then run from the repository root:

```bash
uv lock
uv sync --locked --all-packages --all-groups
```

Review version changes in the root lockfile and run the affected test suites.
Keep pytest and other development-only tools in dependency groups. Maintain one
root lockfile rather than creating member lockfiles.

For a consistency check without installing packages:

```bash
uv lock --check
git diff --check
```
