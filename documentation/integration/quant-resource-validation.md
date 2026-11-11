# Quant runtime resource validation

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


Validated October 24, 2026. Prompts and CUDA source now live under
`quant/src/quant/` and are explicitly included as package data. Their contents
are unchanged from the prior checkout locations.

`quant.data.load_prompts` uses `importlib.resources.files`. The extension loader
uses `files` and `as_file`, keeping the CUDA directory available throughout JIT
compilation. CUDA backend implementation and model behavior are unchanged.

## Distribution checks

Built with:

```bash
uv build --package quant --out-dir /tmp/monorepo-quant-wheel
```

Both source distribution and wheel built successfully; uv built the wheel from
the source distribution. The wheel includes `quant/prompts.json` and
`quant/csrc/bandwidth.cu`, byte-for-byte identical to the original tracked files.

Installed the wheel without dependencies into
`/tmp/monorepo-quant-wheel/installed`, reusing shared-environment dependencies.
From `/tmp`, explicitly imported that installed copy and verified:

- Package location was under the wheel installation, outside the checkout.
- `load_prompts()` loaded all eight prompts with IDs and text.
- The extension loader passed the installed wheel's CUDA file to PyTorch's JIT
  loader. PyTorch's loader was mocked for this resource-path check; it was not
  a compilation test of the installed wheel.

Temporary build and installation artifacts are outside Git and may be removed
after review. The wheel bundles CUDA sources, not a precompiled extension.
Experiment scripts and generated results remain checkout-oriented.

## Existing suite results

All tests ran through the root shared environment using `uv run --locked
--all-packages --all-groups python -m pytest` from `quant/`.

| Suite | Passed | Deselected | Duration | Exit code |
| --- | ---: | ---: | ---: | ---: |
| Default CPU | 52 | 15 | 2.55 s | 0 |
| GPU (`-m gpu`) | 9 | 58 | 33.90 s | 0 |
| Real model (`-m model`) | 6 | 61 | 7.52 s | 0 |

These suites exercised the editable workspace package. GPU checks passed with
the relocated source. No test was skipped in the reported results. The root
`uv lock --check` and `git diff --check` also passed; no lockfile changes were
needed.
