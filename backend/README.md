# Backend

The installable `kernel_backend` package implements the FastAPI API, model loading,
generation helpers, and attention/MLP interventions.

## Layout

- `src/kernel_backend/api.py`: API endpoints; importing this module loads Qwen3-0.6B.
- `src/kernel_backend/schemas.py`: request, generation, intervention, and response schemas.
- `src/kernel_backend/model.py`: model and tokenizer loading.
- `src/kernel_backend/generation.py`: generation and intervention orchestration.
- `src/kernel_backend/interventions.py`: attention head and MLP scaling and noise hooks.
- `src/kernel_backend/errors.py`: custom error classes.
- `tests/`: API endpoint, schema, generation helper, and intervention tests.
- `JSON_examples/`: sample requests and responses.

## Development

From the repository root, install the workspace:

```bash
uv sync --locked --all-packages --all-groups
```

Start the API from either the repository root or `backend/`:

```bash
uv run --locked --all-packages --all-groups python -m uvicorn kernel_backend.api:app
```

Run backend tests from the repository root:

```bash
uv run --locked --all-packages --all-groups python -m pytest -c backend/pyproject.toml backend/tests -v -ra
```
This suite validates API endpoints using mocked model loading and generation, but does not validate production server startup or real-model API inference.

Or from `backend/`:

```bash
uv run --locked --all-packages --all-groups python -m pytest -v -ra
```

- The API currently uses the standard Transformers model loader.
- See the [root README](../README.md) for workspace and quant validation commands.
