# quant
Performs quantization of Qwen3-0.6B weights (the Kernel portion of project)

src/quant/loading.py loads the 16 bit model with `load_model` and loads the INT8 model with `load_int8_model`, and records the baseline datatype with `BASELINE_DTYPE`

src/quant/data.py loads WikiText-2 raw text for the purpose of evaluation, splits it into token chunks, and builds chat template inputs for the prompts in prompts.json

src/quant/quantize.py performs the bulk of the suppression work. it contains symmetric per output channel INT8 quantization and weight error statistics

src/quant/quantized_linear.py consists of `QuantizedLinear`, the drop in replacement for `nn.Linear` (reference backend: FP32 math on the dequantized weight)

src/quant/surgery.py then swaps out any linear layers inside the decoders with `QuantizedLinear`. Note: it never touches `lm_head`, which shares its tensor with the embedding.

src/quant/fidelity.py compares a reference model with a quantized copy. It has per layer activations, sliding window perplexity and logit agreement, and seeded side by side generation.

src/quant/ext.py JIT-builds the CUDA sources in csrc/ (sm_89 only). csrc/bandwidth.cu is the streaming-read kernel used to measure peak DRAM bandwidth.

src/quant/bench/ is the benchmark harness: CUDA-event timing with L2 flushing (timing.py), NVML telemetry in a separate process with run rejection and run-condition metadata (telemetry.py), end-to-end decode timing (e2e.py), and CSV/JSON output to results/bench/ (report.py).

scripts/ contains the experiment scripts. Each one writes its output to results/.

## Usage
```
uv sync
uv run pytest              # fast CPU tests on a tiny random Qwen3
uv run pytest -m gpu       # GPU tests; the first run builds the CUDA extension
uv run pytest -m model     # tests on the real checkpoint (CUDA GPU)
uv run python scripts/inventory.py            # model inventory
uv run python scripts/dtype_check.py          # BF16 vs FP16 baseline decision
uv run python scripts/quant_error_report.py   # INT8 weight error per layer
uv run python scripts/fidelity.py             # FP16 vs INT8-reference fidelity
uv run python scripts/peak_bandwidth.py       # measured peak bandwidth (run before bench_linear)
uv run python scripts/bench_linear.py --backend cublas_fp16   # per-shape batch-1 linear timing
uv run python scripts/bench_e2e.py --precision fp16           # TTFT and decode tok/s at 512/1024/4096
```

Benchmarks expect AC power and the performance power profile (`powerprofilesctl set performance`); the scripts warn otherwise. To lock clocks, run `sudo nvidia-smi -lgc <MHz>` first, pass `--locked-sm-mhz` to bench_e2e.py, and reset with `sudo nvidia-smi -rgc` afterwards.
