# K Track: Findings and Results

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


Status as of 2026-10-14. Covers the development environment (plan task 1.1, section 1) and plan tasks 1.4 (model inventory), 1.5 (baseline dtype), 1.6 (benchmark harness and measured bandwidth), 1.9 (INT8 quantization library), 1.10 (`QuantizedLinear` and model surgery) and 1.11 (fidelity ladder rungs 2–4, FP16 vs INT8-reference). All code is in `quant/`. Raw outputs are in `quant/results/`.

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, sm_89, 20 SMs, 24 MiB L2, 5.64 GiB usable memory |
| Display | Intel iGPU (`prime-select`: on-demand) |
| Driver | 595.91.07, Ubuntu package `nvidia-driver-595-open` (supports CUDA up to 13.2) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0 |
| Python | 3.14.4 |
| torch | 2.14.1+cu130 |
| transformers | 5.18.0 |
| datasets | 5.1.0 |
| Package manager | uv 0.12.23 |

| Prerequisite | Status (checked 2026-10-13) |
|---|---|
| uv | Installed |
| gcc | 15.2.0 |
| Python headers (`python3-dev`) | Installed |
| CUDA toolkit | 13.0.3 (`nvcc` 13.0.88) in `/usr/local/cuda-13.0`; `/usr/local/cuda` points to it |
| Nsight Compute | 2025.3.1 (installed with the toolkit) |
| Nsight Systems | 2025.3.2 (installed with the toolkit) |
| Profiling counters for non-root users | Enabled (`RmProfilingAdminOnly: 0`) |
| Docker | `docker.io` 29.1.3 from Ubuntu, with `docker-buildx` and `docker-compose-v2`. The user is not in the `docker` group, so every Docker command needs `sudo`. |
| NVIDIA Container Toolkit | 1.20.1; the `nvidia` runtime is registered in `/etc/docker/daemon.json` |
| Ninja | 1.13.2, a dependency of `quant/pyproject.toml` (finding 2.12). Not installed system-wide. |
| `CUDA_HOME` / `PATH` in the shell | **Not set.** torch finds the toolkit through `/usr/local/cuda`. `nvcc`, `ncu` and `nsys` are not on the shell's `PATH`. |
| `kernel.perf_event_paranoid` | 4. Nsight Systems cannot sample the CPU as a non-root user. |

### Installation method
- **CUDA toolkit:**
  - Package `cuda-toolkit-13-0`, from NVIDIA's **Ubuntu 24.04** repository. The 26.04 repository only has 13.3 and 13.4.
  - The 24.04 package installs on 26.04: its four missing dependencies are covered by renamed packages (`libasound2t64`, `libglib2.0-0t64`, `libgcc-s1`, `libtinfo6`).
  - nvcc 13.0 accepts host gcc up to version 15.
  - The meta-packages `cuda` and `cuda-13-0` were avoided, because they pull in NVIDIA's driver.
- **Keeping the driver Ubuntu's:**
  - The `cuda-keyring` package installs `/etc/apt/preferences.d/cuda-repository-pin-600`, which ranks NVIDIA's repository (priority 600) above Ubuntu's (500).
  - A second file, `/etc/apt/preferences.d/nvidia-cuda-repo-no-drivers`, sets priority −1 for `nvidia-*`, `libnvidia-*`, `cuda-drivers*`, `nvidia-open*`, `xserver-xorg-video-nvidia*` and `libxnvctrl*` from that repository.
  - Without `libxnvctrl*` in the pin, the next upgrade would have replaced Ubuntu's `libxnvctrl0` 510.47 with NVIDIA's 615.71.
  - A simulated `apt-get dist-upgrade` takes no packages from NVIDIA's repository.
- **Profiling counters:** `/etc/modprobe.d/nvidia-profiling.conf` sets `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, followed by `update-initramfs -u` and a reboot.
- **Container Toolkit:** from NVIDIA's `libnvidia-container` repository, configured with `nvidia-ctk runtime configure --runtime=docker`.

### Verification (non-root, 2026-10-13)

| Check | Result |
|---|---|
| Build a test CUDA kernel with `torch.utils.cpp_extension.load_inline` (`TORCH_CUDA_ARCH_LIST=8.9`) | Builds against `CUDA_HOME=/usr/local/cuda` and torch CUDA 13.0, and the output is correct. The first build takes 51 s. Ninja was supplied by a temporary environment (`uv run --with ninja`). |
| Nsight Compute on the test kernel (a scale of a 64 MiB FP32 vector) | Hardware counters readable. DRAM read 67.11 MB, DRAM write 52.04 MB, 793 µs, 91.8% of peak DRAM throughput. |
| Nsight Systems, CUDA trace | Timeline recorded. The kernel appears at 0.91 ms. |
| GPU inside a container (`sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi`, run by the user 2026-10-14) | The container sees the RTX 4050, driver 595.91.07, CUDA 13.2. It reported a power cap of 82 W. The enforced cap varies between readings (35, 82, 94 and 105 W have been observed), consistent with Dynamic Boost. |

## 2. Findings that affect the plan

1. **GPU inference needs a C toolchain.**
   - torch 2.14 runs the batched outer product in Qwen3's rotary embedding through a Triton kernel.
   - Triton compiles a C stub on first use, which needs both `gcc` and the Python headers. Without them, every forward pass of the real model on the GPU fails.
   - The container image (task 1.2) must include `build-essential` and `python3-dev`.
2. **transformers 5.x changed two names used in the plan.**
   - `rope_theta` is now `config.rope_parameters["rope_theta"]`.
   - `from_pretrained` takes `dtype=` instead of `torch_dtype=`.
3. **The CUDA toolkit version is fixed by torch.** torch 2.14.1 is built against CUDA 13.0 (`nvidia-cuda-runtime 13.0.96`). Kernels built with `cpp_extension` (Phase 2) need toolkit 13.0. It is installed (section 1).
4. **The OS differs from the plan.** It is Ubuntu 26.04; plan task 1.1 assumes 24.04.
   - NVIDIA's 26.04 repository does not carry CUDA 13.0, so the toolkit comes from the 24.04 repository, with an apt pin that keeps Ubuntu's driver (section 1).
   - The container image (task 1.2) can use a 24.04 base, where CUDA 13.0 is supported directly.
5. **Some INT8 scales are below the FP16 normal range.**
   - Three per-channel scales are smaller than 2⁻¹⁴ ≈ 6.10e-5, the smallest normal FP16 value. All three are in `k_proj`:

     | Layer | Smallest scale |
     |---|---|
     | 19 | 5.74e-5 |
     | 21 | 4.09e-5 |
     | 22 | 6.08e-5 |

   - Stored as FP16, these would be subnormal and lose precision. This supports the plan's FP32 scale storage.
   - `QuantizedLinear` keeps its scales in FP32 when the model is cast to another dtype.
   - Plan §5.1 specifies FP16 group scales for INT4. Group scales divide by 7, not 127, so they are about 18× larger. The INT4 scale range should still be checked in Phase 4.
6. **Scale spread within a matrix reaches 11.6×.**
   - The ratio of largest to median row scale reaches 11.6×, in `gate_proj` of layer 27.
   - With per-channel quantization, a large row does not affect the step size of other rows. A per-tensor scale would set the step from the largest row for every row.
7. **The half-step error bound needs a rounding tolerance.**
   - The measured |error| / scale reaches 0.5000018. The theoretical maximum is 0.5.
   - The excess comes from FP32 rounding in computing `q·s − w`, which is at most 127 · 2⁻²³ ≈ 1.5e-5 of the scale.
   - Tests and reports use this tolerance (`FP32_SLACK` in `quant/src/quant/quantize.py`).
8. **The baseline dtype is FP16, with a 2× margin to the plan's threshold.**
   - Layer 2 `down_proj` writes a large activation (up to 8,128) into the residual stream at position 0. It stays there through layer 25.
   - The plan's threshold is 16,376 and the FP16 maximum is 65,504. Details are in section 5.
   - Ablation and quantization can both change this activation. The ablation experiment (task 3.5) needs an inf/NaN check.
9. **`q_norm` increases INT8 error; `k_norm` reduces it.**
   - The local error ratio after / before the norm is 1.15 for `q_norm` and 0.81 for `k_norm`.
   - Plan §4.2 expected both norms to absorb part of the projection error. Section 6 has the details.
10. **Layer 27 cancels most of the position-0 activation, which amplifies relative error there.**
    - Layer 27 shrinks the position-0 residual from about 7,000 to about 2,000. Under INT8-reference the result is 1,576 instead of 2,032, a relative error of 33%.
    - This affects only the prediction at position 0.
    - Kernel error checks (task 3.3) and any per-layer metric must treat position 0 separately or use an absolute tolerance there.
11. **INT8-reference costs 0.31% WikiText-2 perplexity** (19.8865 → 19.9489). That is 17× the BF16↔FP16 difference.
    - No fixed prompt goes from a correct answer to an incorrect one.
    - These numbers are the target the CUDA backend must reproduce.
12. **Building CUDA extensions needs Ninja.**
    - `torch.utils.cpp_extension` fails without it: "Ninja is required to load C++ extensions".
    - It is not in `quant/pyproject.toml`, and it is not installed system-wide.
    - Resolved in task 1.6: `ninja` 1.13.2 is a project dependency, pinned in `uv.lock`, so the container image (task 1.2) gets it too.
13. **The 24 MiB L2 cache can hold the weights a benchmark is timing.**
    - Each INT8 matrix is 1–3 MB, and a whole INT8 decoder layer is 15.7 MB.
    - A benchmark that runs the same matrix repeatedly reads it from L2, not DRAM, and overstates DRAM bandwidth.
    - The test kernel shows the effect: 15 MB of its 67 MB of writes were still in L2 when it finished.
    - Resolved in task 1.6: cold measurements rotate through buffer copies, with at least 2 × L2 read between reuses of a copy (section 7). Warm reads at 8 MiB run at 910 GB/s, 5.0× DRAM.
14. **Some profiling and clock control needs root.**
    - Nsight Systems CPU sampling needs `kernel.perf_event_paranoid` ≤ 2 (currently 4). CUDA tracing works without it, and covers the decode timeline in task 1.7.
    - Clock locking (`nvidia-smi -lgc`, plan §6.2) needs `sudo`, and laptop drivers may refuse it. Not attempted yet.
    - Not needed for kernel-level runs: on the performance profile, every kernel-level block in task 1.6 met the 5% clock rule unlocked. Whether end-to-end runs need it is to be measured in task 1.7.

15. **CUDA event readings cluster at multiples of 1.024 µs on this GPU.**
    - Per-call medians of 10–40 µs kernels would be quantized by 3–10%.
    - Kernel-level timing (task 2.K7) must time batches of calls per event pair. The harness does this (`time_cuda_rotating`).
16. **cuBLAS FP16 GEMV already runs at 86–98% of measured peak bandwidth at the decode shapes.**
    - The 16-bit baseline leaves no slack for the INT8 kernel to exploit beyond the reduced byte count.
17. **Smaller transfers do not reach peak bandwidth.**
    - A cold read reaches 72% of peak at 1 MiB, 85% at 2 MiB and 92% at 4 MiB. The INT8 matrices are 1.1–3.2 MB.
    - The kernel-level INT8 ceiling is therefore 1.67–1.84× per shape and 1.80× per decoder layer, not 2×.
    - With the 16-bit LM head, the kernel-time ceiling per token is 1.50×, against 1.59× in plan Appendix A.
18. **Device-to-device copy measures 9% below streaming read** (166.1 vs 183.4 GB/s).
    - Plan §6.3 allows either as the measured peak. This project uses the streaming read, since GEMV is read-dominated.
19. **Eager FP16 decode runs at 86.7 tok/s (smoke, context 512).**
    - The linears and LM head account for 59% of decode time, from the kernel-time estimate.
    - Launch and Python overhead, plus the other kernels, are the remaining 41%. This limits the end-to-end gain from a faster GEMV, as plan §6.3 anticipated (S2: CUDA graphs).

## 3. Model inventory (task 1.4)

Script: `quant/scripts/inventory.py`. Output: `quant/results/inventory.json`. Every check against plan §4.1 passed.

| Config field | Value |
|---|---|
| num_hidden_layers | 28 |
| hidden_size | 1024 |
| head_dim | 128 |
| num_attention_heads / num_key_value_heads | 16 / 8 |
| intermediate_size | 3072 |
| vocab_size | 151,936 |
| tie_word_embeddings | true |
| attention_bias | false |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-6 |
| max_position_embeddings | 40,960 |
| Stored dtype | bfloat16 |

| Parameter group | Count |
|---|---|
| Total (shared tensor counted once) | 596,049,920 |
| Decoder linears (196 modules) | 440,401,920 |
| Shared embedding / LM head | 155,582,464 |
| RMSNorm weights: 28 × (1024 + 1024 + 128 + 128) + 1024 | 65,536 |

| Module | N | K | Count | Bias |
|---|---|---|---|---|
| self_attn.q_proj | 2048 | 1024 | 28 | none |
| self_attn.k_proj | 1024 | 1024 | 28 | none |
| self_attn.v_proj | 1024 | 1024 | 28 | none |
| self_attn.o_proj | 1024 | 2048 | 28 | none |
| mlp.gate_proj | 3072 | 1024 | 28 | none |
| mlp.up_proj | 3072 | 1024 | 28 | none |
| mlp.down_proj | 1024 | 3072 | 28 | none |

- The only linear outside the decoder layers is `lm_head` (151,936 × 1024).
- `lm_head.weight` and `embed_tokens.weight` share one tensor (same `data_ptr`).
- Every layer has `q_norm` and `k_norm` with weight shape [128].
- Every K (1024, 2048, 3072) is a multiple of 512 and of 128.

## 4. INT8 weight quantization error (task 1.9)

Script: `quant/scripts/quant_error_report.py`. Outputs: `quant/results/int8_weight_error.csv` (one row per linear) and `quant/results/int8_weight_error.md`.

**Method**
- Symmetric, per output channel, round to nearest (half to even).
- Range [-127, 127]; -128 is not used.
- Scales are FP32 and computed from the BF16 checkpoint weights converted to FP32.
- Error is the dequantized weight minus the original weight.

**Overall, 196 matrices**

| Metric | Value |
|---|---|
| Relative Frobenius error, mean | 0.92% |
| Relative Frobenius error, range | 0.82% – 1.20% |
| SQNR, range | 38.40 – 41.77 dB |
| Largest \|error\| / scale | 0.5000018 |
| Clipped weights | 0 |
| Largest \|w\| | 1.234 (layer 27 `up_proj`) |
| Scale range | 4.09e-5 – 9.72e-3 |

**By module type**

| Module | Rel. Frobenius error, mean / max | Min SQNR (dB) | Largest abs. error | Scale max / median, range over layers |
|---|---|---|---|---|
| q_proj | 0.867% / 0.921% | 40.72 | 2.52e-3 | 3.1× – 6.4× |
| k_proj | 0.919% / 1.015% | 39.87 | 3.26e-3 | 2.3× – 10.9× |
| v_proj | 0.893% / 1.048% | 39.59 | 2.00e-3 | 1.6× – 5.0× |
| o_proj | 0.984% / 1.186% | 38.52 | 2.34e-3 | 2.2× – 5.5× |
| gate_proj | 0.874% / 0.968% | 40.29 | 4.55e-3 | 2.7× – 11.6× |
| up_proj | 0.862% / 0.940% | 40.54 | 4.86e-3 | 2.7× – 11.0× |
| down_proj | 1.064% / 1.202% | 38.40 | 2.78e-3 | 3.2× – 6.7× |

**Observations**
- `down_proj` has the highest error of any module type. It accounts for 9 of the 10 matrices with the largest relative error. The other one is layer 11 `o_proj`.
- The per-layer mean error ranges from 0.88% to 0.98%. No single layer stands out. Layers 16–21 are slightly above the rest.
- The highest scale spreads are in the MLP input projections of the last two layers (26, 27) and in layer 16 `k_proj`. Their relative error is not among the highest.
- The largest absolute errors are in `gate_proj` and `up_proj` of the final layers, which have the largest weights.

**Ten largest relative errors**

| Layer | Module | Rel. Frobenius error | SQNR (dB) | Largest \|w\| | Scale max / median |
|---|---|---|---|---|---|
| 19 | down_proj | 1.202% | 38.40 | 0.426 | 3.6× |
| 16 | down_proj | 1.202% | 38.40 | 0.479 | 4.5× |
| 11 | o_proj | 1.186% | 38.52 | 0.395 | 3.9× |
| 18 | down_proj | 1.135% | 38.90 | 0.453 | 4.0× |
| 10 | down_proj | 1.130% | 38.94 | 0.387 | 3.6× |
| 20 | down_proj | 1.120% | 39.01 | 0.535 | 4.4× |
| 14 | down_proj | 1.113% | 39.07 | 0.348 | 3.2× |
| 27 | down_proj | 1.103% | 39.15 | 0.555 | 4.9× |
| 21 | down_proj | 1.092% | 39.24 | 0.602 | 4.7× |
| 2 | down_proj | 1.091% | 39.24 | 0.582 | 5.9× |

These are weight-space errors only. Their effect on activations and perplexity is measured in task 1.11.

## 5. Baseline dtype decision (task 1.5)

Script: `quant/scripts/dtype_check.py`. Outputs: `quant/results/dtype_check.json` (every hook point) and `quant/results/dtype_check.md`.

**Decision: FP16.** It is recorded as `BASELINE_DTYPE` in `quant/src/quant/loading.py`.
- Rule (plan task 1.5): use BF16 if FP16 produces any inf/NaN, or if any |activation| exceeds 65,504 / 4 = 16,376. Otherwise use FP16.
- No condition triggered.
- Models are always loaded in BF16, run through surgery, then cast to FP16.

**Method**
- **Raw text:** 64 non-overlapping 1,024-token chunks of WikiText-2 test, 65,472 scored tokens.
- **Chat prompts:** the 8 prompts in `quant/prompts.json`, each with thinking off and on.
- **Models:** BF16 as loaded, and an FP16 copy cast from it. Both on the GPU, SDPA attention.
- **Hooks:** on every module output and every RMSNorm input (the residual stream). Each records the inf count, the NaN count and the largest finite |x|.

**Results**

| Metric | Value |
|---|---|
| Perplexity, BF16 / FP16 | 24.3598 / 24.3508 |
| inf / NaN values, BF16 and FP16 | 0 |
| KL(BF16 ‖ FP16) per token, mean / max | 8.04e-4 / 0.141 |
| Top-1 agreement | 98.02% |
| Largest \|logit difference\| | 3.10 |
| Largest BF16 \|activation\|, WikiText-2 | 7,456 |
| Largest BF16 \|activation\|, chat prompts | 8,128 |
| Margin to threshold / to FP16 maximum | 2.0× / 8.1× |

**Large activation at position 0**
- **Source:** layer 2 `mlp.down_proj` writes a single large value into the residual stream.
- **Through the layers:** layers 0 and 1 output at most 11. From layer 2 to layer 25 the largest value stays near 7,456. It falls to 7,200 at layer 26 and 2,176 at layer 27.
- **Position:** a separate probe found the peak at position 0 (the first token) in every input. This probe was not saved as a script.
- **Raw text:** the peak is 6,496–7,456 depending on the first token. A 4,096-token context does not raise it.
- **Chat template:** position 0 is always `<|im_start|>`, and the peak is 8,128 for every prompt, with thinking on or off.
- **Why the chat value is fixed:** under causal attention, the activation at position 0 depends only on the first token. With the chat template it is the same for every request.

**Weights cast to FP16**
- **Largest weights:** the largest |weight| is 192.0, in `model.layers.27.post_attention_layernorm.weight`. The five largest tensors by |weight| are all RMSNorm gains. No weight exceeds the FP16 maximum.
- **Precision loss:** 1,186,125 of 596,049,920 weights (0.20%) fall in the FP16 subnormal range, and 571 round to zero.
- **What this affects:** the FP16 baseline, and the 16-bit tensors (embedding, norms) of the quantized models. INT8 weights are quantized from the BF16 checkpoint and are not affected.

**Risks carried forward**
- **Ablation:** an ablation in layers 0–2 could change the position-0 activation. The ablation experiment (task 3.5) needs an inf/NaN check on logits.
- **Quantization:** INT8 quantization of layer 2 `down_proj` changes the value that produces this activation. Task 1.11 measures the effect (section 6).

## 6. INT8-reference fidelity (task 1.11)

Script: `quant/scripts/fidelity.py`. Library: `quant/src/quant/fidelity.py`. Outputs: `quant/results/fidelity_int8_ref.json` (every hook point, all generated tokens) and `quant/results/fidelity_int8_ref.md`.

**Setup**

| Model | How it is built |
|---|---|
| FP16 baseline | Checkpoint loaded in FP16. |
| INT8-reference | Loaded in BF16; all 196 decoder linears quantized; cast to FP16. The reference backend computes in FP32. |
| BF16 | Checkpoint as stored. A yardstick: a change with no quantization at all. |

Runtime on the RTX 4050: rung 2 took 4 s, rung 3 253 s, rung 4 27 s.

### Rung 3: WikiText-2 test perplexity

Sliding window: context 1,024, stride 512, 584 windows, 299,077 scored tokens. Raw text, no chat template.

| Model | Perplexity | Change vs FP16 | KL(FP16 ‖ model), mean / max | Top-1 agreement | Largest \|logit difference\| |
|---|---|---|---|---|---|
| FP16 | 19.8865 | – | – | – | – |
| INT8-reference | 19.9489 | +0.313% | 3.27e-3 / 2.50 | 96.98% | 9.85 |
| BF16 | 19.8901 | +0.018% | 7.99e-4 / 0.459 | 97.99% | 6.23 |

Compared with the BF16 yardstick:
- the INT8 perplexity increase is 17× larger;
- mean KL is 4.1× larger;
- top-1 disagreement is 1.5× larger (3.02% vs 2.02%).

### Rung 2: per-layer activations

**Method**
- **Input:** 8 WikiText-2 test sequences of 512 tokens each.
- **Compounded:** the INT8 model's activations against the FP16 model's, each on its own forward pass. Errors build up through depth.
- **Local:** each INT8 linear applied to the FP16 model's input of that linear. This isolates that layer's own error.
- **Tables:** positions ≥ 1. Position 0 is reported separately.

| Module | Compounded rel. L2, mean / max (layer) | Compounded cosine, mean | Local rel. L2, mean / max (layer) |
|---|---|---|---|
| q_proj | 2.29% / 3.04% (14) | 0.99972 | 0.75% / 1.07% (26) |
| k_proj | 2.85% / 6.84% (12) | 0.99964 | 0.75% / 1.02% (21) |
| v_proj | 4.07% / 7.14% (12) | 0.99914 | 1.28% / 1.80% (20) |
| q_norm | 2.56% / 3.70% (17) | 0.99964 | 0.86% / 1.18% (26) |
| k_norm | 1.95% / 3.49% (13) | 0.99977 | 0.63% / 1.00% (16) |
| o_proj | 5.13% / 6.80% (14) | 0.99852 | 1.04% / 1.44% (13) |
| gate_proj | 2.02% / 2.96% (14) | 0.99977 | 0.61% / 0.75% (26) |
| up_proj | 3.33% / 4.65% (13) | 0.99941 | 1.01% / 1.37% (1) |
| down_proj | 4.59% / 6.57% (13) | 0.99885 | 1.14% / 1.38% (27) |
| Residual stream | 2.61% / 3.20% (16) | 0.99964 | n/a |

**Observations**
- **Local error matches the weight error in size.** Per module type it is 0.61–1.28%, against 0.86–1.06% weight error (section 4).
- **The ranking differs from the weight error.** `v_proj` has the highest local error and `gate_proj` the lowest. In weight space, `down_proj` ranks highest.
- **Compounded error is 2–5× the local error.** It is highest at `o_proj` (5.1%), `down_proj` (4.6%) and `v_proj` (4.1%).
- **Residual stream** (positions ≥ 1):

  | Layer | Rel. L2 |
  |---|---|
  | 0 | 1.23% |
  | 16 (peak) | 3.20% |
  | 26 | 2.39% |
  | 27 | 3.15% |

  The mean per-token cosine is at least 0.99946 at every layer.
- **Isolated tokens have low cosine.** The minimum per-token cosine is 0.17 at `v_proj`, 0.40 at `k_proj` and 0.43 at `o_proj`. The mean cosine of these modules is at least 0.9985. The tokens involved were not identified.

### Q/K norm absorption (positions ≥ 1)

| Projection | Mode | Rel. L2 before norm | After norm | Ratio after / before, mean (range over layers) |
|---|---|---|---|---|
| q_proj | local | 0.745% | 0.860% | 1.15 (0.79 – 1.39) |
| q_proj | compounded | 2.29% | 2.56% | 1.11 (0.73 – 1.36) |
| k_proj | local | 0.754% | 0.625% | 0.81 (0.39 – 1.40) |
| k_proj | compounded | 2.85% | 1.95% | 0.69 (0.35 – 1.07) |

- `k_norm` reduces the relative error, and `q_norm` increases it.
- Plan §4.2 expected the norms to absorb part of the projection error. That holds for `k_norm` only. The cause was not investigated.

### Position 0
- **Layer 2 `down_proj`:** the largest |x| at position 0 is 7,304 in FP16 and 7,284 in INT8 (−0.27%). The relative error at position 0 is 0.73% compounded and 0.63% local.
- **Layers 2–26:** the residual-stream error at position 0 stays between 0.73% and 0.80%.
- **Layer 27:** this layer shrinks the position-0 activation from 7,068 to 2,032 in FP16, but to 1,576 in INT8. The relative error at that point is 33.0%.
- **What layer 27's position-0 output reaches:** only the logits at position 0, which predict token 1. Other positions read layer 27's keys and values, which are computed from layer 26's output (error 0.80% at position 0).
  - In rung 3, only the first window scores the prediction at position 0: 1 of 299,077 tokens.
  - With the chat template, position 0 is `<|im_start|>`, and its prediction is never used.

### Rung 4: generations

8 prompts, non-thinking chat template, seeded sampling (temperature 0.7, top-p 0.8, top-k 20), 128 new tokens.

| Prompt | First token where INT8 differs from FP16 | First token where BF16 differs from FP16 |
|---|---|---|
| capital | identical | identical |
| arithmetic | 3 | identical |
| code | identical | identical |
| instruction | identical | identical |
| explanation | 17 | 17 |
| summary | 27 | identical |
| translation | identical | identical |
| reasoning | 101 | identical |

- INT8 output differs from FP16 on 4 of 8 prompts. BF16 differs on 1 of 8 (explanation), at the same token as INT8.
- **Correctness:** no prompt goes from a correct answer to an incorrect one.
  - **arithmetic:** both answer 391, which is correct. INT8 orders the two partial products differently.
  - **reasoning:** FP16 and INT8 both answer "5:65 pm" (the correct answer is 6:05 pm). The error belongs to the baseline model. INT8 differs only in the wording of the last sentence.
  - **explanation:** after token 17, every continuation contains factual errors. The INT8 one describes a heater warming the interior.
  - **summary:** both are correct. INT8 ends the sentence earlier.

### Use as the CUDA backend's reference
- The CUDA backend (tasks 2.K6 and 3.3) is checked against INT8-reference, not against FP16.
- Its perplexity should match 19.9489 within noise.
- Its per-layer outputs should match INT8-reference within the kernel tolerance.
- At layer 27, position 0 needs an absolute tolerance or a separate check (finding 2.10).

## 7. Benchmark harness and measured bandwidth (task 1.6)

**Code:**
- the harness: `quant/src/quant/bench/`;
- the streaming-read kernel: `quant/csrc/bandwidth.cu`;
- the JIT loader for CUDA sources: `quant/src/quant/ext.py`;
- the scripts: `quant/scripts/peak_bandwidth.py`, `bench_linear.py` and `bench_e2e.py`.

All outputs go to `quant/results/bench/`.

**Run conditions** for every number below:
- AC power, performance power profile, clocks not locked;
- 2026-10-14;
- the enforced power limit, read from NVML, was 102 W.

### Method
- **Timing:**
  - Times come from CUDA events.
  - A 20 ms sleep kernel is queued before each timed loop. The CPU enqueues every iteration while the GPU sleeps, so the GPU then runs them back to back, and short kernels are timed without CPU launch overhead.
  - **Short kernels are timed in batches.** CUDA event readings on this GPU cluster at multiples of 1.024 µs; for example, 61 of 200 readings of a 4 µs sleep were exactly 4.096 µs. Per-call medians of 10–40 µs kernels would therefore be quantized by 3–10%. Each sample instead times R back-to-back calls between one event pair and divides by R (R = 9–49).
- **Cache state:**
  - Cold measurements rotate through R copies of the data. Between two reads of the same copy, at least 2 × L2 (48 MiB) of other copies is read, so every call reads DRAM.
  - Warm measurements reuse one buffer.
  - The harness also has an L2 flusher (`L2Flusher`) for single-call timing. It flushes by *reading* a 48 MiB buffer: a write flush leaves dirty lines, whose write-back adds DRAM traffic inside the next timed call.
  - All buffers are filled with random data, because zero-filled buffers can read faster on GPUs with data compression.
- **Telemetry and rejection:**
  - A separate process samples NVML every 100 ms: SM clock, memory clock, temperature, power, and the clock-event (throttle) reasons. Being separate, it doesn't compete with the benchmark for Python's GIL.
  - The sampler discards its first power reading: the first reading after NVML starts up has come back invalid (590 W).
  - A block is rejected if its SM clock spread, (max − min) / median, exceeds 5%; the block is then retried, up to 3 attempts.
  - Each block runs for at least 0.5 s and starts with one untimed priming call, so its telemetry window holds its own samples. Without these, short blocks were judged by idle readings: the first `F.linear` call spends about 0.5 s initializing cuBLAS on the CPU while the GPU sits idle.
- **Run conditions recorded with every result:** AC status, power profile, driver, versions, git commit, and a clock-lock check (for when you lock clocks with `sudo nvidia-smi -lgc` and pass `--locked-sm-mhz`).
- **End-to-end runner:**
  - **Prompt:** a non-thinking chat prompt of exactly the requested length, filled with WikiText-2 test tokens.
  - **Loop:** a hand-written decode loop with KV cache; end-of-sequence is ignored; seeded sampling on the GPU with the model-card settings.
  - **Time to first token (TTFT):** CUDA-event time from the start of prefill to the first sampled token.
  - **Decode speed:** decode tokens/s excludes the first token.
  - **Per-precision processes:** one precision per process (plan §6.4).
  - **Determinism:** runs with the same seed generate identical tokens, and this is checked on every run.

### Measured peak bandwidth (`peak_bandwidth.{md,json}`)

| Method, 1 GiB buffer | GB/s | % of NVML theoretical (187.8 GB/s) | % of spec sheet (192 GB/s) |
|---|---|---|---|
| Streaming read (custom kernel) | **183.4** | 97.6% | 95.5% |
| `torch.sum` | 183.3 | 97.6% | 95.5% |
| Device-to-device copy (read + write) | 166.1 | 88.5% | 86.5% |

- **The efficiency denominator is the streaming read, 183.4 GB/s.**
- **The theoretical figure:** NVML reports a maximum memory clock of 7,825 MHz, which on the 96-bit bus gives 187.8 GB/s. The spec sheet's 192 GB/s corresponds to 16 Gbps memory.
- **Copy vs read:** copy is 9% slower than read, so the plan's "copy or streaming read" choice changes the denominator by 9%. GEMV is read-dominated, so read is the right denominator.
- **Stability:** 54 of 54 configurations were accepted, with no retries. The SM clock was 2,355–2,520 MHz, the memory clock 7,825 MHz throughout, with no power-cap or thermal flags.

**Bandwidth by buffer size** (median GB/s). Cold means every call reads DRAM; warm means the same buffer every call.

| Size (MiB) | Read, cold | Read, warm | Copy, cold | torch.sum, cold |
|---|---|---|---|---|
| 1 | 132.7 | 235.6 | 174.1 | 100.2 |
| 2 | 156.6 | 409.4 | 170.7 | 128.6 |
| 4 | 168.5 | 641.5 | 167.6 | 150.8 |
| 8 | 175.9 | 910.2 | 167.1 | 164.8 |
| 16 | 179.1 | 829.6 | 166.3 | 172.9 |
| 32 | 180.7 | 180.7 | 166.3 | 171.6 |
| 64 | 181.5 | 181.5 | 166.3 | 179.6 |
| 1024 | 183.4 | 183.4 | 166.1 | 183.3 |

- **L2 bandwidth:** warm reads reach 910 GB/s at 8 MiB, 5.0× DRAM. At 32 MiB they fall back to the DRAM rate, consistent with the 24 MiB L2.
- **Small transfers:** cold reads of small buffers stay below peak, because each call has a fixed start-up and drain cost. A cold read reaches 72% of peak at 1 MiB, 85% at 2 MiB and 92% at 4 MiB.

### Batch-1 linear layers (`bench_linear_<backend>.{md,json}`)

Every call reads its weights from DRAM (rotating copies). Times are medians over at least 0.5 s of batched calls.
- **Effective GB/s:** the plan §6.3 bytes model (weights + scales + x + y) divided by time.
- **Size-matched read:** the streaming-read kernel on a buffer with the same byte count, timed the same way.

| Shape | cuBLAS FP16 (µs) | GB/s | % of peak | % of size-matched read | INT8-reference (µs) | % of peak |
|---|---|---|---|---|---|---|
| q_proj (2048 × 1024) | 24.7 | 169.8 | 92.6% | 100.7% | 49.7 | 23.2% |
| k_proj, v_proj (1024 × 1024) | 13.3 | 157.8 | 86.1% | 100.6% | 30.3 | 19.0% |
| o_proj (1024 × 2048) | 24.2 | 173.7 | 94.7% | 102.9% | 50.4 | 22.8% |
| gate_proj, up_proj (3072 × 1024) | 35.0 | 179.8 | 98.0% | 103.9% | 72.8 | 23.7% |
| down_proj (1024 × 3072) | 35.4 | 177.9 | 97.0% | 102.8% | 73.7 | 23.4% |

- **cuBLAS FP16 is already bandwidth-saturated:** it reaches 86–98% of measured peak at these shapes.
- **cuBLAS exceeds the size-matched read by up to 4%.** A sweep of the read kernel's launch configuration (20–1,280 blocks × 128–1,024 threads) gains 1–5% at these sizes and still stays below cuBLAS at 6.3 MB. So the size-matched read is a reference, not a ceiling. cuBLAS presumably uses loads tuned better for DRAM; that is worth examining for the Phase 2 kernels.
- **INT8-reference is 2.0–2.3× slower than cuBLAS FP16.** It converts the whole INT8 weight to an FP32 copy on every call, then runs an FP32 matmul. That moves several times the bytes the model counts. It is a correctness reference, not a speed baseline.

**Kernel-level INT8 ceiling.** The time for an INT8 kernel that reads its bytes at the size-matched read rate:

| Shape | INT8 bytes | Size-matched read (GB/s) | INT8 ceiling (µs) | Speedup over cuBLAS FP16 |
|---|---|---|---|---|
| q_proj | 2,111,488 | 156.7 | 13.5 | 1.84× |
| k_proj, v_proj | 1,056,768 | 132.7 | 8.0 | 1.67× |
| o_proj | 2,107,392 | 156.9 | 13.4 | 1.80× |
| gate_proj, up_proj | 3,166,208 | 164.3 | 19.3 | 1.82× |
| down_proj | 3,158,016 | 164.4 | 19.2 | 1.84× |

- **Per decoder layer:** all 7 linears take 181.0 µs in FP16 against an INT8 ceiling of 100.6 µs, a 1.80× ratio rather than the 2.0× that halving the bytes would suggest. Smaller transfers run at lower bandwidth.
- **Per token, kernel time only:**
  - The 196 linears plus the 16-bit LM head (1.70 ms at peak read) take 6.77 ms in FP16, which is 148 tok/s.
  - At the INT8 ceiling they take 4.51 ms, which is 222 tok/s.
  - The ratio is 1.50×, against 1.59× in plan Appendix A, which assumes every transfer runs at full bandwidth.

### End-to-end smoke tests (not baselines)

Context 512, 128 generated tokens, 5 runs (INT8: 5 runs). These only check the runner; the baselines are task 1.7.

| Precision | Decode tok/s | TTFT (ms) | Peak allocated (GiB) | Allocated after loading (GiB) |
|---|---|---|---|---|
| FP16 | 86.7 | 35.6 | 1.202 | 1.110 |
| INT8-reference | 61.8 | 96.8 | 0.821 | 0.730 |

- **FP16 decode time:** 11.5 ms per token. The kernel-time estimate for the linears and the LM head is 6.77 ms, 59% of that. The remainder is the other kernels (attention, norms, rotary, sampling) plus Python and launch overhead. Task 1.7 will break this down with Nsight Systems.
- **Repeatability:** every run was accepted, and the generated tokens were identical across runs.
- **Profile:** a smoke run on the balanced profile gave 86.1 tok/s, within 1% of the performance profile.

## 8. Code and tests

| File | Purpose | State |
|---|---|---|
| `src/quant/loading.py` | Loads the 16-bit model (`load_model`) and the INT8 model (`load_int8_model`); `BASELINE_DTYPE` | Done |
| `src/quant/data.py` | WikiText-2 raw text, token chunks, chat-template inputs, prompt set | Done |
| `prompts.json` | Fixed prompt set (8 prompts) for task 1.5 and fidelity rung 4 | Done |
| `src/quant/quantize.py` | INT8 per-channel quantize / dequantize, error stats | Done |
| `src/quant/quantized_linear.py` | `QuantizedLinear`; reference backend computes `s · (x @ qᵀ)` in FP32 | Done; `"cuda"` backend raises until Phase 2 |
| `src/quant/surgery.py` | Replaces decoder linears in place; skips `lm_head`; verifies the shared tensor afterwards | Done |
| `scripts/inventory.py` | Task 1.4 | Run |
| `scripts/quant_error_report.py` | Task 1.9 report | Run |
| `scripts/dtype_check.py` | Task 1.5 | Run |
| `src/quant/fidelity.py` | Activation comparison (compounded and local, position 0 separate), sliding-window perplexity and logit agreement, seeded side-by-side generation | Done |
| `scripts/fidelity.py` | Task 1.11 | Run |
| `src/quant/ext.py`, `csrc/bandwidth.cu` | JIT build of CUDA sources (sm_89, `-lineinfo`); streaming-read kernel | Done |
| `src/quant/bench/` | Harness: timing (`timing.py`), NVML telemetry and run conditions (`telemetry.py`, `_nvml_worker.py`), rejection and retry (`runner.py`), end-to-end decode (`e2e.py`), statistics (`stats.py`), CSV/JSON (`report.py`) | Done |
| `scripts/peak_bandwidth.py` | Task 1.6: measured peak bandwidth | Run |
| `scripts/bench_linear.py` | Task 1.6, runner for 2.K7: batch-1 linears at the five shapes | Run (`cublas_fp16`, `int8_ref`) |
| `scripts/bench_e2e.py` | Task 1.6, runner for 1.7 and 3.4: TTFT and decode tok/s | Smoke runs only |

**Tests**
- **Fast suite, 52 tests, CPU:** all pass. Coverage:
  - quantization invariants;
  - `QuantizedLinear` matches the FP32 formula and stays within the analytic error bound;
  - the scales stay FP32 when the model is cast;
  - hooks see the returned tensor;
  - surgery replaces every target layer and leaves module paths, `lm_head` and the shared tensor unchanged;
  - `generate()` runs after surgery;
  - FP16 input is refused;
  - fidelity: identical models show zero error, every hook point is covered, position 0 is counted separately, sliding windows score each token once, perplexity matches a hand computation, and identical models generate identical tokens;
  - harness: statistics, the bytes model, clock-stability rejection, telemetry windows, the clock-lock check, the rotation count, block extension and priming, seeded top-k/top-p sampling, and CSV/JSON output.
- **GPU suite (`pytest -m gpu`, 9 tests):** all pass.
  - The event timer scales with sleep length.
  - The read kernel's checksum matches, including byte tails.
  - L2 flushing and rotation both produce cold reads.
  - Batched times resolve below the 1.024 µs clustering.
  - The NVML sampler returns plausible samples.
- **Real-model suite (`pytest -m model`, 6 tests, GPU):** all pass.
  - Surgery replaces all 196 linears.
  - The shared tensor holds after surgery.
  - The INT8 model answers "What is the capital of France?" with "Paris" (seeded sampling, BF16 activations).
  - Chat prompts are built at exactly 512, 1024 and 4096 tokens.

## 9. Open items

1. Environment, remaining:
   - optional: add `CUDA_HOME` and `PATH` to `~/.bashrc`;
   - optional: set `kernel.perf_event_paranoid=2` for Nsight Systems CPU sampling.
2. Task 1.7: FP16 end-to-end baselines at 512, 1024 and 4096 tokens of context (50 runs each, `bench_e2e.py`), and an Nsight Systems timeline of one decode step.
   - Record the rejection rate; if the 5% rule rejects many runs, try a clock lock (finding 2.14).
   - Then Phase 2 kernels, benchmarked with `bench_linear.py` (finding 2.15).
3. Phase 2 (2.K3/2.K4): compare the INT8 kernel's load pattern with cuBLAS, which exceeds the size-matched streaming read by up to 4% (section 7).
4. Task 3.3: compare the CUDA backend against INT8-reference; handle position 0 at layer 27 separately (finding 2.10).
5. Task 3.5: add an inf/NaN check on logits to the ablation experiment (finding 2.8).
6. Optional: identify the tokens with low per-token cosine under INT8 (minimum 0.17 at `v_proj`, section 6), and the cause of error amplification by `q_norm` (finding 2.9).
7. Phase 4: check the range of INT4 FP16 group scales against the FP16 normal range (finding 2.5).
8. No work has been committed to git yet.
