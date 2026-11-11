# Shared workspace validation

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


Recorded October 24, 2026, on `integration/uv-workspace`. This validates the first
dependency migration against [the independent baselines](independent-baselines.md).
Results were reported and confirmed by the developer; tests were not rerun when
writing this document.

## Workspace configuration

- Root `pyproject.toml` declares `backend` and `quant` as uv workspace members.
- Root `.python-version` selects Python 3.14.
- Root `uv.lock` resolves both projects together: 116 packages.
- The shared environment is `monorepo/.venv`.
- Both member manifests and source layouts remain intact.
- Environment setup used `uv sync --locked --all-packages --all-groups` from the
  repository root.

## Dependency reconciliation

The initial unconstrained root resolution selected 122 packages and introduced
unrelated upgrades. Temporary constraints derived from the original lockfiles
restored the baseline versions. For the five packages with different original
versions, quant's baseline versions were selected and resolved successfully.

| Package | Backend baseline | Quant baseline | Shared lockfile |
| --- | --- | --- | --- |
| filelock | 4.0.10 | 4.0.12 | 4.0.12 |
| fsspec | 2026.9.0 | 2026.7.0 | 2026.7.0 |
| hf-xet | 1.6.0 | 1.7.0 | 1.7.0 |
| iniconfig | 2.3.0 | 2.3.1 | 2.3.1 |
| typer | 0.27.2 | 0.27.3 | 0.27.3 |

All other original package versions were preserved. No packages were added beyond
the union of the original lockfiles, and no original packages were removed.
Torch remains 2.14.1, Transformers 5.18.0, and pytest 9.1.1.

Temporary constraints were removed from the root manifest. A subsequent `uv lock`
retained the selected versions, and `uv lock --check` passed without those
constraints. Both original lockfiles remained unchanged during reconciliation.

## Validation results

All four shared-environment suites matched their independent baseline pass counts.

| Suite | Independent passed | Shared passed | Shared deselected | Evidence |
| --- | ---: | ---: | --- | --- |
| Backend | 29 | 29 | Not separately supplied | Developer confirmation |
| Quant CPU | 52 | 52 | Not separately supplied | Developer confirmation |
| Quant GPU | 9 | 9 | 58 | Supplied rerun output; exit code 0 |
| Quant model | 6 | 6 | Not separately supplied | Developer confirmation |

The successful GPU rerun reported Linux, Python 3.14.4, pytest 9.1.1, the root
`monorepo/.venv/bin/python`, and 9 passed / 58 deselected in 31.58 seconds.
Full shared-environment output and exit codes for the other three suites were not
pasted into the conversation; the developer confirmed their pass counts.

## Test launch issue and correction

The first GPU invocation used `../.venv/bin/python` directly from `quant/`.
It produced Python prefix warnings and failed the first CUDA extension test with
`Ninja is required to load C++ extensions`; the other eight tests passed.

The suspected cause was that selecting Python directly did not put the shared
environment's executable directory on PATH. Repeating the suite through `uv run`
passed all nine tests and removed the prefix warnings without dependency changes.
Keep both the initial failure log and successful rerun as evidence.

Use the following commands for future checks. Each block works from any starting
directory on this machine, preserves the member's pytest configuration and working
directory, and stops if changing directories fails.

```bash
# Backend
(
  cd /home/james17/projects/capstone/monorepo/backend &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra
)

# Quant CPU (default configuration excludes gpu and model markers)
(
  cd /home/james17/projects/capstone/monorepo/quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra
)

# Quant GPU
(
  cd /home/james17/projects/capstone/monorepo/quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra -m gpu
)

# Quant real model
(
  cd /home/james17/projects/capstone/monorepo/quant &&
  uv run --locked --all-packages --all-groups python -m pytest -v -ra -m model
)
```

These commands are the recommended launch pattern after resolving the GPU issue;
they are not a claim that every supplied result was obtained with this exact form.

## Log retention

Independent logs were originally stored in `/tmp/monorepo-baseline.hFHOm2`.
Shared-environment logs were directed to a directory created by
`mktemp -d /tmp/monorepo-workspace.XXXXXX`; its exact suffix was not supplied.
The recommended durable parent directory is
`/home/james17/monorepo-validation/`. The developer indicated the log-preservation
step looked good, but exact saved paths and file contents have not been inspected
for this record. Confirm these before removing temporary copies.

## Scope and remaining migration steps

The existing test suites pass with the shared dependency resolution. This does not
establish API startup, a real `/generate` request, wheel resource completeness, or
performance equivalence; those checks are outside this migration's validation.

After reviewing this record, retire the nested `backend/uv.lock` and `quant/uv.lock`
while retaining both member manifests and the root lockfile. Then run
`uv lock --check` and `git diff --check` from the repository root and review the
complete change before committing. This document does not assert those final
cleanup steps or the migration commit have occurred.
