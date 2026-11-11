# K Track Guide: Work to Date and Next Steps

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


Status as of 2026-10-20. The K track is the kernel and quantization half of [the revision 3 project plan](pre_trained_fast_lab_plan_v3.md). This guide explains what was planned, how it was built, what was found, and what comes next.
- Full result tables are in [the K-track findings](k_track_findings.md).
- Raw outputs are in `quant/results/`.
- Section numbers like §5.1 refer to the plan.

## 1. Project context

- **Research question:** does weight quantization change which model components are important (plan §2)?
- **The two halves of the project:**
  1. The **K track** quantizes the linear layers of Qwen3-0.6B (INT8, later INT4) and writes CUDA kernels that run decoding directly on the quantized weights.
  2. The **U track** builds the ablation environment and web UI that measure component importance.
- **Hardware:** one RTX 4050 Laptop GPU (sm_89, 6 GB). It produces every headline number.
- **Scope so far:** all work has been on the K track. `backend/` (the U-track FastAPI server) has not been modified.

## 2. Deviations from plan v3

| Plan | Actual | Reason |
|---|---|---|
| Ubuntu 24.04 (task 1.1) | Ubuntu 26.04 | The machine's OS. The CUDA toolkit comes from NVIDIA's 24.04 repository, because the 26.04 repository has no CUDA 13.0. |
| `transformers` ≥ 4.51 | 5.18.0 on Python 3.14 | Pinned to match `backend/uv.lock`. Two API renames: `rope_theta` moved into `config.rope_parameters`, and `torch_dtype=` is now `dtype=`. |
| Baseline dtype FP16 or BF16 (task 1.5) | **FP16** | Selected by the plan's rule (section 5.2). |
| University compute (task 1.8, §6.1) | **Dropped** | GPU hours may be rented later, both for a second, newer device (task 4.3) and to host the web demo. The container image (task 1.2) is how the setup moves to a rented machine. |
| Tasks in numeric order | Tasks that need only plain PyTorch first (1.4, 1.5, 1.9, 1.10, 1.11), then 1.1 and 1.6 | No CUDA toolkit was installed at the start. No dependency was violated. |
| Measured peak: copy or streaming read (§6.3) | Streaming read | GEMV is read-dominated; copy measures 9% lower. |
| Kernel timing method: not specified | Rotating buffer copies, timed in batches | Required by the L2 size and by the event timer's granularity (section 5.7). |
| Run rejection (§6.2) | Adds a minimum block length (0.5 s) and an untimed priming call | Without them, short blocks were judged on idle telemetry (section 5.7). |
| Docker | Docker without the `docker` group | Membership in that group is root-equivalent. Every Docker command needs `sudo`, so the operator runs them. |

## 3. Repository layout

```
quant/                       separate package; backend/ will depend on it later
  pyproject.toml, uv.lock    pinned: torch 2.14.1 (CUDA 13.0), transformers 5.18.0, datasets, ninja, nvidia-ml-py
  prompts.json               8 fixed chat prompts (tasks 1.5 and 1.11)
  csrc/bandwidth.cu          streaming-read kernel (peak bandwidth)
  src/quant/
    loading.py               load_model, load_int8_model, BASELINE_DTYPE = float16
    data.py                  WikiText-2 loader, token chunks, chat-template inputs
    quantize.py              INT8 per-channel quantize/dequantize, error stats, FP32_SLACK
    quantized_linear.py      QuantizedLinear (reference backend: FP32 math)
    surgery.py               quantize_model_, verify_tied_embeddings
    fidelity.py              activation comparison, sliding-window eval, side-by-side generation
    ext.py                   JIT build of csrc/ sources (sm_89, -O3, -lineinfo)
    bench/                   benchmark harness (section 5.7)
  scripts/                   one script per task; each writes to results/
  tests/                     pytest; markers: model (real checkpoint), gpu (CUDA, may build extensions)
  results/                   small JSON / CSV / Markdown outputs, committed
documentation/               plan, findings, this guide (gitignored)
```

**Repository state:**
- **Committed:** commit `91e4de8` ("Initial model quantization") contains everything through task 1.11.
- **Uncommitted:** the task 1.6 work: `csrc/`, `src/quant/ext.py`, `src/quant/bench/`, three scripts, `tests/test_bench.py`, `results/bench/`, and the `pyproject.toml` / `uv.lock` changes.
- **Tracked by mistake:** `quant/src/quant.egg-info/` (a build artifact).
- **Not in the repository:** `documentation/` is gitignored, so a collaborator who clones the repository does not receive these documents.

## 4. Environment

| Component | Version / setting |
|---|---|
| Driver | 595.91.07, Ubuntu package `nvidia-driver-595-open`; supports CUDA up to 13.2 |
| CUDA toolkit | 13.0.3 in `/usr/local/cuda-13.0`; `/usr/local/cuda` points to it |
| Nsight Compute / Nsight Systems | 2025.3.1 / 2025.3.2, installed with the toolkit |
| Profiling counters | readable without root (`NVreg_RestrictProfilingToAdminUsers=0`) |
| Docker / NVIDIA Container Toolkit | 29.1.3 / 1.20.1; a GPU is visible inside containers |
| Python tooling | uv 0.12.23, in `~/.local/bin` |

**Things that will catch a new engineer:**
- **C toolchain needed for inference.** torch 2.14 runs the batched outer product in Qwen3's rotary embedding through a Triton kernel, which compiles a C stub on first use. Any forward pass on the GPU therefore needs `gcc` and `python3-dev`.
- **Driver protection in apt.** The `cuda-keyring` package ranks NVIDIA's repository above Ubuntu's (`/etc/apt/preferences.d/cuda-repository-pin-600`). A second pin, `/etc/apt/preferences.d/nvidia-cuda-repo-no-drivers`, blocks NVIDIA's driver packages so the Ubuntu driver stays in place.
  - Never install the `cuda` or `cuda-13-0` meta-packages: they pull in NVIDIA's driver.
  - After any repository change, run `apt-get -s dist-upgrade` and check that no package would come from `developer.download.nvidia.com`.
- **Toolkit not on PATH.** Nothing puts `nvcc`, `ncu` or `nsys` on `PATH`; prefix commands with `PATH=/usr/local/cuda-13.0/bin:$PATH`.
- **No root-free CPU sampling.** `kernel.perf_event_paranoid` is 4, so Nsight Systems cannot sample the CPU without root. CUDA tracing works.
- **Clock locking needs root.** `sudo nvidia-smi -lgc <MHz>` has not been attempted yet.
- **Bad first NVML power reading.** The first NVML power reading after startup can be invalid (590 W was observed). The sampler discards it.
- **Run conditions for benchmarks:** AC power and `powerprofilesctl set performance`. The scripts warn when either is missing.

## 5. Completed tasks

### 5.1 Task 1.4: model inventory
- **Planned:** confirm the plan's model tables (§4.1), the tensor shared between `lm_head` and the embedding, and the `q_norm` / `k_norm` modules.
- **Built:** `scripts/inventory.py`. It loads the model in BF16 on the CPU, asserts every value against the plan, exits non-zero on any mismatch, and writes `results/inventory.json`.
- **Results:** every value matches.

  | Item | Value |
  |---|---|
  | Total parameters | 596,049,920 |
  | Decoder linears | 196 modules, 440,401,920 parameters, no biases |
  | Shared embedding / LM head | 155,582,464 |
  | RMSNorm weights | 65,536 |
  | `rope_theta` | 1,000,000 |

- **Implications:**
  - The kernels need only five (N, K) shapes: 2048×1024, 1024×1024, 1024×2048, 3072×1024 and 1024×3072.
  - Every K is a multiple of 512, so the model never exercises a K tail. Tail handling must be tested with synthetic shapes.

### 5.2 Task 1.5: baseline dtype
- **Planned:** run BF16 and FP16 on at least 50k tokens. Use BF16 if FP16 produces any inf/NaN or any activation within 4× of 65,504; otherwise FP16.
- **Built:** `scripts/dtype_check.py`.
  - Both models are resident on the GPU at once.
  - Hooks record the inf count, NaN count and largest finite |x| at every module output and at every RMSNorm input (the residual stream).
  - Statistics are computed on the GPU in 16M-element slices. A single int64 sum over the logits ran out of memory.
  - Inputs: 64 non-overlapping 1,024-token WikiText-2 chunks, plus the 8 chat prompts with thinking on and off.
  - It also checks what casting the weights to FP16 does to them.
- **Results:**
  - **Overflow:** no inf or NaN in either dtype.
  - **Largest activation:** 7,456 on raw text and 8,128 on chat prompts, against a threshold of 16,376. The peak is always at position 0. Layer 2 `down_proj` writes it into the residual stream, and it persists through layer 25.
  - **Perplexity and logits:** BF16 24.3598, FP16 24.3508. Mean KL 8.0e-4, top-1 agreement 98.02%.
  - **Weights:** the largest |weight| is 192, an RMSNorm gain. 0.20% of weights become subnormal in FP16 and 571 round to zero.
- **Decision:** FP16, recorded as `BASELINE_DTYPE`. Models are always loaded in BF16, run through surgery, then cast.
- **Implications:**
  - The margin is 2× to the threshold and 8× to overflow. Ablations near layers 0–2 could change the position-0 activation, so the ablation experiment (task 3.5) needs an inf/NaN check.
  - Kernels target FP16 activations first.

### 5.3 Task 1.9: quantization library
- **Planned:** symmetric per-channel INT8 with round-to-nearest, quantized from the BF16 weights, with tests on the round-trip error.
- **Built:** `quantize.py`.
  - Math is in FP32: scale = row max |w| / 127; all-zero rows get scale 1; values are clamped to ±127 (-128 is never used).
  - It refuses FP16 input.
  - `weight_error_stats` reports the error statistics, and `FP32_SLACK` (127 · 2⁻²³) is the rounding tolerance on the half-step bound.
  - `scripts/quant_error_report.py` runs it on the real weights.
- **Results:**
  - **Error:** relative Frobenius error 0.92% on average (range 0.82–1.20%), SQNR 38.4–41.8 dB, no clipping.
  - **Worst module type:** `down_proj`, which has 9 of the 10 largest errors.
  - **Scales:** three `k_proj` scales fall below the smallest normal FP16 value (minimum 4.09e-5). Within one matrix, row scales spread up to 11.6×.
- **Implications:**
  - Scales must stay FP32; `QuantizedLinear` enforces this.
  - The INT4 group scales, specified as FP16, need the same range check in Phase 4.

### 5.4 Task 1.10: `QuantizedLinear` and model surgery
- **Planned:**
  - `QuantizedLinear`: an INT8 weight, FP32 scales and an optional bias, with a reference backend in plain PyTorch;
  - surgery that leaves `lm_head` alone and verifies the shared tensor afterwards.
- **Built:**
  - **The reference backend** computes `F.linear(x.float(), q.float()) * scales (+ bias)` and casts the result back to the input dtype. That is the exact math in plan §5.1 (FP32 accumulation, scale applied at the end), and it is the ground truth the CUDA kernel is tested against.
  - **An `_apply` override** keeps the scales in FP32 when the model is cast to FP16 or BF16.
  - **Backend `"cuda"`** raises `NotImplementedError` until Phase 2.
  - **`quantize_model_`** replaces every `nn.Linear` under `model.model.layers` in place, keeping module paths unchanged so hooks work identically. It refuses FP16 models before replacing anything, then verifies the shared tensor.
  - **`load_int8_model`** wraps loading, surgery and the cast.
- **Results:** all 196 linears are replaced, the shared tensor survives, and the INT8 model generates coherent text.
- **Implications:** the reference backend is slow. It is 2.0–2.3× slower than cuBLAS FP16 per layer, and decodes at 61.8 vs 86.7 tok/s end to end. It is a correctness reference, not a speed baseline.

### 5.5 Task 1.11: fidelity ladder, rungs 2–4
- **Planned:**
  - rung 2: per-layer activation error, including before and after `q_norm` / `k_norm`;
  - rung 3: WikiText-2 perplexity;
  - rung 4: fixed-prompt generation.
- **Built:** `fidelity.py` and `scripts/fidelity.py`.
  - **`compare_activations`** measures error two ways:
    - compounded: the INT8 model against the FP16 model, each on its own forward pass;
    - local: each INT8 layer applied to the FP16 model's input to that layer.

    Position 0 is reported separately, because its large activation would dominate any average.
  - **`sliding_window_eval`:** context 1,024, stride 512, every token scored once, plus KL divergence and top-1 agreement against FP16.
  - **`generate_side_by_side`:** same seed for every model, recording the first token where outputs differ.
  - **BF16** is included as a yardstick: a change that involves no quantization.
- **Results:**
  - **Perplexity:** FP16 19.8865; INT8 19.9489 (+0.313%); BF16 19.8901 (+0.018%). Mean KL for INT8 is 3.27e-3, and top-1 agreement 96.98%.
  - **Local error:** 0.61–1.28% per module type.
  - **Compounded error:** 2–5× the local error. In the residual stream it peaks at 3.20% at layer 16.
  - **Q/K norms:** `q_norm` increases the error (ratio 1.15) and `k_norm` reduces it (0.81). The plan (§4.2) expected both to absorb error.
  - **Position 0:** the layer 2 `down_proj` value is 7,304 in FP16 and 7,284 in INT8. Layer 27 cancels most of that activation, and the cancellation turns a small difference into a 33% relative error. It affects only the prediction at position 0.
  - **Generations:** INT8 differs from FP16 on 4 of 8 prompts; no prompt goes from a correct answer to an incorrect one.
- **Implications:**
  - These INT8-reference numbers are the target the CUDA backend must reproduce.
  - Error checks at layer 27 need special handling at position 0.

### 5.6 Task 1.1: development environment
- **Planned:** CUDA toolkit, Nsight, profiling counters for non-root users, and the display on the integrated GPU.
- **Done:** see section 4. Verification:
  - a test CUDA extension built against torch's CUDA 13.0 and returned correct results;
  - Nsight Compute read DRAM counters without root;
  - Nsight Systems recorded a CUDA trace;
  - a container saw the GPU.

### 5.7 Task 1.6: benchmark harness and measured bandwidth
- **Planned:**
  - CUDA-event timing, warmup, median and IQR over N ≥ 50 runs;
  - clock, temperature and power logged at 100 ms, with runs rejected beyond a 5% SM clock spread;
  - CSV output;
  - a measured-peak-bandwidth test;
  - context length and generated-token count as parameters.
- **Built:**

  | File | Role |
  |---|---|
  | `csrc/bandwidth.cu` | Streaming-read kernel: 16-byte loads, four in flight per thread, XOR-accumulated, one word written per block. |
  | `bench/timing.py` | `time_cuda` (one call per event pair, with a queued sleep ahead of the loop); `time_cuda_rotating` (batches of calls on rotating copies); `rotation_count`; `L2Flusher`. |
  | `bench/telemetry.py`, `bench/_nvml_worker.py` | NVML sampler in a separate process; telemetry windows; the stability rule; the clock-lock check; `environment()`. |
  | `bench/runner.py` | `timed_block`: priming call, block of at least 0.5 s, retry up to 3 times. |
  | `bench/e2e.py` | Exact-length chat prompt, sampling on the GPU, hand-written decode loop that measures TTFT and decode tokens/s. |
  | `bench/stats.py`, `bench/report.py` | Median and IQR, the bytes model, CSV/JSON output. |
  | `scripts/peak_bandwidth.py` | Read, copy and `torch.sum` at 1 MiB–1 GiB, warm and cold. |
  | `scripts/bench_linear.py` | The five shapes, cold, compared with peak and with a read of the same size. Backends: `cublas_fp16`, `int8_ref`. |
  | `scripts/bench_e2e.py` | One precision per process. Runs are rejected and repeated; the script checks that tokens are identical across runs. |

- **Measurement problems found and fixed:**
  1. **Launch overhead.** A short kernel timed alone included Python's launch time, because the GPU's queue ran empty. Fix: queue a 20 ms sleep kernel before the timed loop.
  2. **Timer granularity.** CUDA event readings cluster at multiples of 1.024 µs, which quantizes 10–40 µs kernels by 3–10%. Fix: time R calls per event pair and divide by R.
  3. **L2 cache.** The 24 MiB L2 holds whole matrices, so repeated calls on one matrix read from L2. Fix: rotate through copies spaced at least 2 × L2 apart. A copy benchmark also exceeded the theoretical bandwidth, because its last writes were still in L2 when timing stopped; the rotation fixed that too.
  4. **Idle telemetry.** Blocks shorter than the 100 ms sampling interval, and blocks dominated by cuBLAS initialization, were judged on idle readings (210 MHz). Fix: a 0.5 s minimum block and an untimed priming call.
  5. **Flush and buffer contents.** The `L2Flusher` flushes by reading, not writing, because write-back of dirty lines would add traffic inside the next timed region. Benchmark buffers hold random data, because zero-filled data may be compressed.
- **Results** (performance profile, clocks unlocked):
  - **Peak read:** 183.4 GB/s, 97.6% of the 187.8 GB/s theoretical figure (2 × 7,825 MHz × 96 bit / 8). Copy reaches 166.1 GB/s.
  - **L2:** warm reads reach 910 GB/s at 8 MiB.
  - **Small transfers:** cold reads reach 72%, 85% and 92% of peak at 1, 2 and 4 MiB.
  - **cuBLAS FP16:** 86–98% of peak at the five shapes. It exceeds the size-matched read by up to 4%.
  - **INT8 ceiling** (INT8 bytes at the read rate measured for that size): 1.67–1.84× faster than cuBLAS FP16 per shape, 1.80× per decoder layer. Per token, counting kernel time only and including the 16-bit LM head, it is 1.50×; plan Appendix A estimated 1.59×.
  - **Stability:** every block passed the 5% rule unlocked. The SM clock ran at 2,355–2,520 MHz and the memory clock held at 7,825 MHz.
  - **End-to-end smoke tests** (context 512, 128 tokens): FP16 86.7 tok/s with 35.6 ms TTFT; INT8-reference 61.8 tok/s.
  - **Decode time split:** the linear layers and LM head account for about 59% of FP16 decode time, estimated from kernel times.
- **Implications:**
  - **Kernel efficiency** must be judged against the read rate at that size, and against the 16-bit control kernel, not only against the 1 GiB peak.
  - **End-to-end gain:** if everything outside the linear layers stays the same, the end-to-end speedup from INT8 at its ceiling is about 1.24×. Launch overhead (plan stretch item S2: CUDA graphs) becomes a first-order concern.
  - **Kernel benchmarks** must use `time_cuda_rotating`.

## 6. Conventions and invariants
- **Loading:** load BF16 → run surgery → cast to `BASELINE_DTYPE`. Never quantize from FP16.
- **Scales** are FP32 everywhere.
- **Module paths** do not change under surgery; hooks are registered by path.
- **Position 0** is reported separately in every activation statistic.
- **Peak bandwidth** means the streaming read at 1 GiB, 183.4 GB/s, read by later scripts from `results/bench/peak_bandwidth.json`.
- **Kernel timing:** cold, batched and rotating, gated by telemetry; median and IQR of at least 50 samples.
- **End-to-end runs:**
  - non-thinking chat prompts at exact lengths;
  - end-of-sequence ignored;
  - seeded sampling with temperature 0.7, top-k 20, top-p 0.8;
  - one precision per process.
- **Every result file** carries its run conditions (`environment()`).

## 7. Tests

| Suite | Command | Count | Covers |
|---|---|---|---|
| CPU (default) | `uv run pytest` | 52 | quantization invariants, `QuantizedLinear`, surgery on a tiny random Qwen3, fidelity functions, harness logic |
| GPU | `uv run pytest -m gpu` | 9 | event timer, read-kernel checksum, L2 flush and rotation, timer batching, NVML sampler |
| Real model | `uv run pytest -m model` | 6 | surgery on Qwen3-0.6B, the shared tensor, generation, exact-length prompts |

All 67 pass. Run them from `quant/`. The first GPU run builds the CUDA extension, which takes about 30–50 s.

## 8. Status against the plan (K track)

| Task | Status |
|---|---|
| 1.1 environment, 1.4 inventory, 1.5 dtype, 1.6 harness, 1.9 quantization, 1.10 `QuantizedLinear`, 1.11 fidelity | Done |
| 1.7 16-bit baselines | **Next** |
| 1.2 container image | Not started |
| 1.8 university compute | Dropped |
| 1.3 API contract, 1.14 prompt builder, 1.15 vertical slice | Joint K+U tasks; not started |
| 2.K1–2.K7 kernels | Not started; the harness and test patterns are ready |
| 3.1, 3.3, 3.4 integration | Not started |
| 4.1 INT4, 4.3 other GPUs | Not started; 4.3 will use rented GPUs |

**Phase 1 exit criteria:**
- The dtype decision and the Nsight counter check are met.
- The 16-bit baselines are task 1.7.
- "INT8 reference generates through the real API" needs task 1.15, which needs the server.

## 9. Next steps

### 9.1 Housekeeping
1. **Commit the task 1.6 work.** Add `quant/src/quant.egg-info/` and `quant/.pytest_cache/` to `.gitignore`, and remove the egg-info from the index with `git rm -r --cached quant/src/quant.egg-info`.
2. **Decide whether `documentation/` should stay gitignored.** Another engineer cannot read these documents from the repository.
3. **Optional:** add `CUDA_HOME=/usr/local/cuda-13.0` and its `bin/` to `PATH` in `~/.bashrc`.

### 9.2 Task 1.7: 16-bit baselines (next)
**Goal:** FP16 time to first token, decode tokens/s and peak memory at 512, 1,024 and 4,096 tokens of context, with variance, plus an Nsight Systems timeline of one decode step. Phase 3 compares against these numbers.

**Procedure:**
1. Use AC power and the performance profile, with clocks unlocked.
2. Run `uv run python scripts/bench_e2e.py --precision fp16` (defaults: contexts 512, 1,024 and 4,096; 128 generated tokens; 50 accepted runs; 3 warmup runs). Expected time: under 15 minutes.
3. **Check the rejection rate for each context length.**
   - If more than about 20% of runs are rejected, lock the clock at or below the lowest sustained SM clock seen so far (2,355 MHz) with `sudo nvidia-smi -lgc 2355,2355`.
   - Rerun with `--locked-sm-mhz 2355`, then reset with `sudo nvidia-smi -rgc`.
   - Record whether the lock held; the JSON has a `clock_lock` field for this.
4. **Run `--precision int8_ref` with the same settings.** It is the INT8-reference row for task 3.4, and it runs in a separate process as plan §6.4 requires.
5. **Check memory:** compare `weights_allocated_bytes` with the analytic figures (FP16 1.19 GB, i.e. 1.11 GiB; INT8 0.78 GB, i.e. 0.73 GiB, including the shared embedding). Record the peak allocated memory per context length.
6. **Timeline of a decode step:**
   - Add NVTX ranges (`torch.cuda.nvtx.range_push` / `range_pop`) around prefill and around each decode step. Put them in a dedicated script, `scripts/profile_decode.py`, so the timed runner is not affected.
   - Profile with:
     ```
     PATH=/usr/local/cuda-13.0/bin:$PATH nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
       -o results/nsys/decode_fp16_ctx512 uv run python scripts/profile_decode.py --ctx 512 --gen 32
     ```
   - Summarize with `nsys stats --report cuda_gpu_kern_sum,nvtx_sum`.
   - For one step, report: kernel count; total GPU kernel time against the step's wall time; GEMV time (cuBLAS `gemv` kernels) against attention, norms, RoPE and elementwise kernels; and idle gaps.
   - Repeat at context 4,096.
7. **Compare with the estimate:** the kernel-time model says the linear layers and LM head take 6.77 ms of the 11.53 ms per token at context 512.

**Acceptance:**
- 50 accepted runs per context length, with identical tokens across runs;
- median and IQR reported;
- the Nsight timeline saved and summarized;
- a new section in `k_track_findings.md`;
- the clock-lock decision recorded.

**What it tells us:** the measured share of decode time spent in GEMV. That sets the realistic end-to-end target for task 3.4, and decides how early to work on CUDA graphs (S2).

### 9.3 Task 1.2: container image
Rented GPUs make this the way the setup moves to other machines.
1. **Base image:** `nvidia/cuda:13.0.x-devel-ubuntu24.04`. The devel variant provides `nvcc` for extension builds, and 24.04 supports CUDA 13.0 directly.
2. **Install:** `build-essential`, then uv. Let uv provide Python 3.14 (`uv python install 3.14`); its Python builds include the headers Triton needs (confirm on the first build). Copy `quant/pyproject.toml` and `uv.lock`, run `uv sync --frozen`, then copy the source.
3. **Model weights:** mount the Hugging Face cache as a volume rather than building the weights into the image.
4. **Test on the laptop:**
   ```
   sudo docker build -t fastlab-quant quant/
   sudo docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface fastlab-quant \
     uv run pytest -m "gpu or model"
   ```
   Then run `scripts/inventory.py`.
5. **Test on the first rented GPU:** run the same commands plus `peak_bandwidth.py`.
6. **Architecture setting:** `ext.py` builds for sm_89 by default (`TORCH_CUDA_ARCH_LIST` is set only if unset). Pass `-e TORCH_CUDA_ARCH_LIST=<arch>` on any other GPU: 8.0 for A100, 9.0 for H100, and 8.9 for the RTX 4090 and L4.

### 9.4 Phase 2: INT8 GEMV kernels
**Shared setup:**
- Sources go in `quant/csrc/gemv.cu`, loaded through `ext.load_extension`.
- **Kernel interface:** `int8_gemv(x[1,K] half, qweight[N,K] int8, scales[N] float32, bias?) -> y[1,N] half`.
- **Activation type:** templated, as plan §5.2 specifies. Only FP16 is needed (section 5.2); a BF16 instantiation is kept for other models.
- **Backend:** `QuantizedLinear` gets `backend="cuda"`, which uses the kernel when T = 1.
- **The 16-bit control kernel** shares the structure of each version but reads FP16 weights. It isolates the effect of the narrower weights from implementation quality.

| Step | Work | Acceptance |
|---|---|---|
| 2.K1 v0 | One thread per output row, scalar loads; matching 16-bit control kernel. | Matches the reference on all five shapes. |
| 2.K2 tests | New `tests/test_gemv.py` (marker `gpu`), compared with the reference backend (FP32 math). Under `compute-sanitizer` (memcheck; racecheck once shared memory is used). | Relative error < 1e-3. Clean sanitizer runs. |
| 2.K3 v1 | One warp per row. Each lane loads 16 int8 values per iteration (a `uint4`), so K = 1024 takes 2 iterations. FP32 accumulation, warp-shuffle reduction, scale (and bias) applied at the end. | DRAM throughput close to the size-matched read rate, checked in Nsight Compute. |
| 2.K4 v2 | Input vector x staged in shared memory (at most 6 KB). Several rows per warp or block, so more loads are in flight. K-tail path. Launch shape as template parameters. Same changes to the control kernel. | Clean on synthetic tails. Efficiency at least that of v1. |
| 2.K5 autotuner | For each shape: sweep threads per block and rows per block, timed with `time_cuda_rotating`; best config saved to `results/autotune/<device>.json`. | A config file for the RTX 4050, loaded at runtime. |
| 2.K6 prefill and dispatch | For T > 1: dequantize into a preallocated 6.3 MB FP16 scratch buffer, then `F.linear`. Dispatch on T. | Fidelity tests pass on both paths. |
| 2.K7 benchmarks | Add `bench_linear.py` backends `int8_v0`…`int8_v3` and `fp16_control_v*`. Report % of peak, % of the size-matched read, and speedup over cuBLAS FP16. | v2 or v3 reaches at least 90% of the size-matched read, and approaches the 1.67–1.84× ceilings. |

**Design inputs from measurements:**
- **Fixed per-call cost.** Each kernel call pays about 2 µs to start up and drain. That caps the 1 MB k_proj / v_proj INT8 kernels at 1.67×. Fusing k and v into one launch would help, but it changes the module structure (plan stretch item S3); defer it as the plan says.
- **cuBLAS's memory access.** cuBLAS exceeds the size-matched read by up to 4%. Profile its GEMV kernel with Nsight Compute and compare: load width, `ld.global.nc`, L2 prefetch hints, loads in flight per thread, and grid size relative to the 20 SMs. Use the Source page with `-lineinfo` for our kernels.
- **Profiling commands:**
  - `ncu -k regex:gemv --set full`;
  - key metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed`, achieved occupancy, and warp stall reasons.
- **Timing:** time only with `time_cuda_rotating`. Single-call event times on this GPU are quantized.

### 9.5 Phase 3: integration
- **3.1 Serve the CUDA backend.** This needs the server. Joint tasks 1.3 (API contract), 1.14 (prompt builder: `data.chat_inputs` exists, but stripping thinking content from history does not) and 1.15 (vertical slice) come first.
  - `backend/` must add `quant` as a path dependency and load `load_int8_model`.
  - `backend/` currently ignores intervention settings and returns hardcoded latencies, so it needs U-track work first.
- **3.3 Fidelity on the CUDA backend.** Add an option to `scripts/fidelity.py` to compare the CUDA backend against INT8-reference. Acceptance:
  - perplexity equal to INT8-reference's 19.9489 within noise (the evaluation is deterministic, so expect a difference under 0.01%);
  - per-layer relative error under 1e-3, with position 0 at layer 27 checked against an absolute tolerance.
- **3.4 End-to-end benchmarks.**
  - Add precision `int8_cuda` to `bench_e2e.py`, and compare FP16, INT8-reference and INT8-CUDA at three context lengths.
  - Use Nsight Systems timelines to explain the gap between kernel-level and end-to-end speedup.
  - The expected end-to-end gain without CUDA graphs is about 1.2×. If the 1.7 timeline confirms that overhead dominates, raise S2 (CUDA graphs with persistent ablation masks) in priority.

### 9.6 Phase 4
- **4.1 INT4:**
  - group size 128 with FP16 group scales, two values per byte;
  - check every group scale against the smallest normal FP16 value, since some INT8 scales already fall below it (section 5.3);
  - new kernel variant, then the fidelity ladder and the importance experiment at INT4 (task 4.2, joint).
  - Plan reference [2] predicts measurable quality loss for round-to-nearest INT4 on Qwen3.
- **4.3 Rented GPUs:**
  - Use the container image. Rerun `peak_bandwidth.py`, the autotuner and `bench_linear.py`.
  - The harness reads the L2 size and memory clock from the device, so rotation counts and the theoretical bandwidth adapt automatically.
  - Re-check the 1.024 µs event clustering and NVML behavior on each new GPU.
  - Report in a separate table; never mix with laptop numbers (plan §6.1).
  - A same-architecture card (sm_89, for example an RTX 4090 or L4) gives the most direct comparison.

### 9.7 Hosting on a rented GPU
- **One instance can host the web demo.**
  - The server processes one generation at a time (plan §5.6).
  - FP16 plus INT8 resident copies, with the KV cache at 4,096 tokens, need about 3.3 GB (plan Appendix A), which fits on any rented GPU.
- **Telemetry differs on a data-centre GPU** (fixed power caps, no Dynamic Boost). The demo's telemetry panel should report the host device.
- **Headline numbers stay on the laptop**, which is the primary device.

### 9.8 Open decisions
1. Clock locking for end-to-end runs: decided in task 1.7 from the rejection rate.
2. Sharing `documentation/`: currently gitignored.
3. Who builds joint tasks 1.3, 1.14 and 1.15, which block the Phase 1 exit criterion and task 3.1.
4. Which rented GPU to use, which sets `TORCH_CUDA_ARCH_LIST` and how directly it compares with the laptop.

## 10. Reproducing the results
From `quant/`, on AC power with the performance profile:
```
uv sync
uv run pytest && uv run pytest -m gpu && uv run pytest -m model
uv run python scripts/inventory.py              # 1.4
uv run python scripts/dtype_check.py            # 1.5, about 1 min
uv run python scripts/quant_error_report.py     # 1.9
uv run python scripts/fidelity.py               # 1.11, about 5 min
uv run python scripts/peak_bandwidth.py         # 1.6, run before bench_linear
uv run python scripts/bench_linear.py --backend cublas_fp16
uv run python scripts/bench_linear.py --backend int8_ref
uv run python scripts/bench_e2e.py --precision fp16 --ctx 512 --runs 5 --tag smoke
```

## 11. Glossary
- **GEMV / GEMM:** matrix–vector and matrix–matrix products. Decode runs GEMVs (memory-bound); prefill runs GEMMs (compute-bound).
- **SM:** streaming multiprocessor. The RTX 4050 has 20.
- **Warp:** 32 threads that execute together. **Coalescing:** the hardware merges neighbouring addresses read by a warp into a few wide transactions.
- **L2 cache:** 24 MiB, shared by all SMs. Warm data is read from L2; cold data from DRAM.
- **Rotation:** cycling through buffer copies spaced at least 2 × L2 apart, so every call reads DRAM.
- **Size-matched read:** the streaming-read kernel on a buffer of the same byte count as a layer. It gives the achievable rate at that transfer size.
- **TTFT:** time to first token. It covers prefill and sampling the first token.
- **Compounded / local error:** error accumulated through the network, versus one layer's own error on identical input.
- **Position 0:** the first token of a sequence, which carries a residual activation of about 7,000–8,000 in this model.
