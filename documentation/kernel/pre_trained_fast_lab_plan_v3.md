# Pre Trained Fast Lab: Technical Project Plan (Revision 3)

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


Weight-only quantized LLM inference with custom CUDA kernels, and an interactive environment for ablation experiments on full-precision and quantized models.

**Revision history**
- Revision 2: GEMV-first kernel design, symmetric per-channel quantization, vertical-slice phasing, head-importance experiment.
- Revision 3: The project uses Qwen3-0.6B as its only model, replacing Qwen1.5-0.5B-Chat for development and TinyLlama-1.1B-Chat for the demo. TinyLlama becomes an optional replication target in Phase 4. Sections 4–7, the risks, and Appendix A are updated for Qwen3's architecture: head dim 128, GQA, Q/K normalization, shared embedding and LM-head weights, BF16 storage, and thinking mode.

---

## 1. Summary

The project has two components.

1. **Inference engine.** The linear-layer weights of Qwen3-0.6B are quantized to INT8, with INT4 as an extension. Custom CUDA kernels compute the matrix-vector products used in autoregressive decoding directly on the quantized weights. Activations stay at 16-bit precision (W8A16).
2. **Intervention environment.** A web application serves the model for interactive generation. Users can ablate attention heads, KV groups, and MLP components during the forward pass, switch between precisions, and compare outputs and hardware telemetry side by side.

The components share one experimental question (Section 2). The inference engine produces quantized models. The intervention environment is the instrument used to measure how quantization changes model behavior.

## 2. Research Question

**Does weight quantization change which model components are important?**

For each attention head and each KV group, measure the increase in negative log-likelihood (ΔNLL) on a held-out corpus when that component is ablated. Do this for the 16-bit baseline, INT8, and INT4. Compare the per-component importance vectors across precisions.

Expected outputs:
- Rank correlation (Spearman ρ) of component importance between the 16-bit baseline and each quantized model.
- Overlap of the top-k most important components across precisions.
- The components whose importance changes most under quantization, and whether those components are concentrated by layer or by component type (query heads, KV groups, MLP).

Rationale: Without a shared question, the kernel work and the interpretability work are two independent projects that share a server. The question also removes a confound in the demo. Degradation observed after ablating a quantized model is a combination of ablation effect and quantization effect. It can only be interpreted against the 16-bit model under the same ablation.

## 3. Scope

### 3.1 In scope
- Symmetric per-output-channel INT8 weight quantization (round-to-nearest).
- A custom CUDA GEMV kernel for the decode path, with fused dequantization, templated on activation type (FP16 or BF16).
- A correct but unoptimized prefill path for multi-token inputs.
- Symmetric group-wise INT4 weight quantization (group size 128) and a corresponding GEMV kernel (Phase 4).
- Ablation of query heads, KV groups, and MLP components, in both zero and mean modes.
- A web UI with streaming generation, a thinking-mode toggle, ablation controls, a precision toggle, a comparison view, next-token probability display, an on-demand attention heatmap, and telemetry.
- The head-importance experiment from Section 2.
- Optional (Phase 4): the same experiment on TinyLlama-1.1B-Chat, to test whether the result holds on a second architecture.

### 3.2 Out of scope
- **Activation quantization (W8A8).** LLM activations contain outlier channels. Quantizing them requires dedicated methods (LLM.int8(), SmoothQuant), and that is a separate project.
- **Calibration-based quantizers (GPTQ, AWQ).** Round-to-nearest is the baseline. Any quality loss it causes is a reported result.
- **Beating cuBLAS on compute-bound GEMM.** The decode-path claim rests on reduced memory traffic, not on matching vendor GEMM tuning.
- **Batched multi-user decoding.** The server processes one generation at a time.
- **Training or fine-tuning.**

### 3.3 Components kept at 16-bit precision
- The shared embedding / LM-head tensor.
- All RMSNorm weights, including the per-head `q_norm` and `k_norm`.
- Rotary embeddings.
- The KV cache.

Rationale: The quantized components are the seven linear layers per block (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), which hold 440M of the model's 596M parameters. The embedding lookup reads one row per token, so quantizing it gains nothing. The LM head is sensitive to quantization, and in Qwen3 it is the same tensor as the embedding. Its bandwidth cost is analyzed in Appendix A, and quantizing it is stretch item S4.

## 4. Model

### 4.1 Selection

**Qwen3-0.6B** (`Qwen/Qwen3-0.6B`) is the only model used for development, the demo, and the experiment.

| Property | Value |
|---|---|
| Layers | 28 |
| Hidden size | 1024 |
| Head dim | 128 (set explicitly; not hidden size ÷ heads) |
| Query heads / KV heads | 16 / 8 (GQA, 2 query heads per KV group) |
| MLP intermediate size | 3072 |
| Vocabulary | 151,936 |
| QKV bias | None (`attention_bias: false`) |
| Q/K normalization | Per-head RMSNorm on queries and keys, applied before rotary embedding |
| Embedding and LM head | One shared tensor (`tie_word_embeddings: true`) |
| Stored dtype | BF16 |
| Parameters | about 596M total, about 440M non-embedding |
| 16-bit weights in memory | about 1.19 GB |
| Minimum `transformers` version | 4.51.0 (older versions raise `KeyError: 'qwen3'`) |

**Linear layer shapes (N = output features, K = input features):**

| Module | N | K |
|---|---|---|
| `q_proj` | 2048 | 1024 |
| `k_proj`, `v_proj` | 1024 | 1024 |
| `o_proj` | 1024 | 2048 |
| `gate_proj`, `up_proj` | 3072 | 1024 |
| `down_proj` | 1024 | 3072 |

**Rationale:**
- **Chat quality.** Coherent baseline output is what makes degradation under ablation visible to a viewer. Qwen3-0.6B is a stronger chat model than either earlier candidate.
- **Current architecture.** GQA, Q/K normalization, and shared embeddings are standard in current models. Results on this architecture are more relevant than results on older designs.
- **Kernel-friendly shapes.** Every K dimension (1024, 2048, 3072) is a multiple of 512, which is 32 lanes × 16 bytes per vector load, and a multiple of 128, the INT4 group size.
- **Scope.** One model means one set of kernel shapes, one set of mean-ablation statistics, and one experiment sweep.

**Excluded models:**
- **Qwen1.5-0.5B-Chat.** Its only advantage over Qwen3-0.6B was plain multi-head attention, which gives a one-to-one mapping between heads and KV slices. It has an older architecture and weaker chat output.
- **TinyLlama-1.1B-Chat.** It has the larger bandwidth-bound speedup ceiling (1.88× vs. 1.59× at INT8; see Appendix A), but maintaining a second model roughly doubles the work for Phases 1–3. It is kept as an optional replication target (task 4.4).
- **Pythia-1B.** Its main advantage is published training checkpoints for interpretability research, which this project does not use. Its base-model output is not suited to a chat demo.

The model-surgery code (task 1.10) stays architecture-agnostic, so any of these can be added later.

### 4.2 Architecture implications

| Property | Consequence | Handled in |
|---|---|---|
| `head_dim` = 128 ≠ 1024 / 16 | `q_proj` output and `o_proj` input are 2048 wide. Code that computes head slices as `hidden_size // num_heads` selects the wrong slice. Always read `config.head_dim`. | 5.5, task 1.12 |
| GQA, 16 / 8 heads | Query-head ablation and KV-group ablation are distinct operations. Ablating one KV group affects 2 query heads. | 5.5, 5.7 |
| Q/K normalization | `q_norm` and `k_norm` stay at 16-bit. Normalization removes scale error in the q and k projections but not direction error, so INT8 error in those layers may be partly absorbed. Measure error both before and after the norms. The norm modules are additional hook points. | 3.3, 5.5, 6.5 |
| Shared embedding / LM head | Model surgery must skip `lm_head`; replacing it breaks the sharing. All precision copies share one 16-bit embedding tensor. Quantizing the LM head (S4) requires a separate copy of the tensor. | 5.4, 5.6, S4 |
| BF16 storage | FP16 has a much smaller range (maximum 65,504) and may overflow on this model's activations. The 16-bit baseline dtype is chosen in task 1.5. The kernel is templated on activation type. | 5.1, 5.2, task 1.5 |
| Thinking mode on by default | The chat template emits a `<think>...</think>` block unless `enable_thinking=False`. The model card advises against greedy decoding in thinking mode, because of repetition and quality loss. Thinking output makes context long, which reduces the quantization speedup (Appendix A). | 5.6, 5.7, 6.3 |
| No QKV bias | The kernel's bias epilogue is optional and unused for this model. It is kept for other architectures. | 5.2 |
| Sensitivity to low-bit quantization | Published results show Qwen3 degrades more than LLaMA3 under low-bit post-training quantization, most sharply at 3 bits and below, while holding up at moderate bit widths [2]. INT8 is expected to be safe. Round-to-nearest INT4 at 0.6B parameters may degrade measurably. | 6.6, task 4.2, Section 9 |

## 5. System Design

### 5.1 Quantization scheme

**INT8, symmetric, per output channel.** For a weight matrix W of shape [N, K]:

```
s_n    = max_k |W[n,k]| / 127
q[n,k] = clamp(round(W[n,k] / s_n), -127, 127)      stored as int8
y_n    = s_n * Σ_k q[n,k] * x_k   (+ b_n, if bias present)
```

The scale factors out of the inner sum. Dequantization therefore costs one multiply per output element, not one per weight. The inner loop is an int8-to-float conversion and a fused multiply-add.

- **Symmetric over asymmetric.** Transformer weights are approximately zero-centered, so a zero-point improves accuracy very little. It would also add a correction term involving Σ_k x_k to every output.
- **Per-channel over per-tensor.** Weight magnitudes vary widely across output channels. A single per-tensor scale spends most of the INT8 range on a few large-magnitude rows.
- **Scale storage.** FP32, one [N] vector per matrix, 1.4 MB in total for the model. FP32 removes one source of numerical error during debugging, and its bandwidth cost is under 0.2%.
- **Source weights.** Quantize from the original BF16 checkpoint weights, converted to FP32. Never quantize from an FP16 copy, which may already have been rounded or clipped.
- **Activation dtype.** FP16 or BF16, whichever is chosen in task 1.5. It applies to the baseline and to all quantized models, so every comparison uses one activation dtype.

**INT4, symmetric, group-wise (Phase 4).** One FP16 scale per contiguous group of 128 input elements in each row. Values range over [-7, 7], and two values are packed per byte. Per-group partial sums are scaled and then accumulated. All K dimensions (1024, 2048, 3072) divide evenly by 128.

### 5.2 Kernel design: decode path

At batch size 1, each linear layer in decoding is a matrix-vector product (GEMV). Each weight is read once and used for one multiply-add, which gives about 1 FLOP per byte for INT8 weights. The operation is bound by memory bandwidth. Its runtime is set by the bytes read, and reducing weight bytes reduces runtime roughly in proportion (Appendix A).

The kernel is templated on activation type (`half` or `__nv_bfloat16`); sm_89 supports both natively. It is developed in stages. Each stage is benchmarked and profiled before the next begins.

| Version | Change | Purpose |
|---|---|---|
| v0 | One thread per output row, scalar loads. | Correctness baseline and a working build pipeline. |
| v1 | One warp per output row, 128-bit vectorized weight loads (16 int8 values per lane per iteration), FP32 accumulation, warp-shuffle reduction, per-row scale and optional bias applied in the epilogue. | Coalesced weight reads. This is the main bandwidth improvement. |
| v2 | Activation vector staged in shared memory, multiple rows per block, K-tail handling, block size and rows per block as template parameters. | Reuse of the activation vector across rows. The K-tail path is not exercised by Qwen3 (all K are multiples of 512) but is required for other architectures and is tested with synthetic shapes. |
| v3 | Autotuned launch configuration per (N, K) shape and per device. | Hardware-specific tuning without code changes. |

- **Activation vector size.** The largest is 3072 elements, or 6 KB, which fits in shared memory on all target devices.
- **Rows per warp.** With K = 1024, one warp covers a row in 2 iterations. Assigning several rows per warp or per block keeps enough loads in flight to saturate bandwidth. This is covered by the v2 and v3 tuning parameters.
- **Int8 conversion.** Start with a plain cast. Optimize with byte-permutation conversion only if Nsight Compute shows the kernel is not saturating DRAM throughput.
- **16-bit control kernel.** Build a GEMV kernel with the same structure that reads 16-bit weights. Comparing the INT8 kernel against it attributes any speedup to reduced memory traffic rather than to differences in implementation quality. Also compare against `torch.nn.functional.linear` (cuBLAS) as the external reference.

### 5.3 Kernel design: prefill path

Prefill (processing the prompt, T > 1 tokens) is a matrix-matrix product and is compute-bound. The initial implementation dequantizes the weight into a preallocated 16-bit scratch buffer and calls cuBLAS through `torch.nn.functional.linear`.

- **Scratch buffer.** One buffer sized to the largest quantized matrix: 3072 × 1024 × 2 bytes = 6.3 MB.
- **Rationale.** Prefill runs once per request. Its cost is the dequantization pass, which is small relative to the GEMM. A fused INT8 tensor-core GEMM is a stretch item (S1).

### 5.4 PyTorch integration

- **Build.** Use `torch.utils.cpp_extension`. Use JIT `load()` during development and an ahead-of-time `setup.py` build for the server. Compile for the target architecture only (`TORCH_CUDA_ARCH_LIST="8.9"`). This shortens builds and gives Nsight Compute unambiguous SASS.
- **Dependencies.** Pin exact versions of `transformers` (≥ 4.51.0), PyTorch, and CUDA in the container image (task 1.2).
- **`QuantizedLinear(nn.Module)`.**
  - Buffers: `qweight` (int8, [N, K], row-major), `scales` ([N], FP32), and an optional `bias`.
  - Forward pass: flattens the input to [T, K] and dispatches on T. T = 1 uses the GEMV kernel. T > 1 uses the prefill path.
  - A `backend` attribute selects between `"reference"` (pure PyTorch: dequantize, then `F.linear`) and `"cuda"`.
- **Model surgery.**
  - A function replaces the named `nn.Linear` submodules inside decoder layers with `QuantizedLinear`, in place, and leaves the module hierarchy and attribute names unchanged. Hooks registered by module path therefore work identically on 16-bit and quantized models.
  - The function never touches `lm_head`, which shares its tensor with `embed_tokens`.
  - After surgery, verify the sharing still holds: `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`.
- **Execution mode.** Eager only. `torch.compile` and CUDA graphs are excluded from the base design because hooks change dynamically. See S2 for a graph-compatible alternative.

### 5.5 Intervention engine

All slice widths come from `config.head_dim` (128). Nothing derives them from `hidden_size`.

- **Query-head ablation.**
  - A `register_forward_pre_hook` on `o_proj` multiplies its input by a mask. The input has shape [B, T, 16 × 128 = 2048].
  - Setting the slice `[h·128 : (h+1)·128]` to zero ablates query head h.
  - This is exactly equivalent to removing head h's output, and it does not depend on the attention implementation (eager, SDPA, or flash).
- **KV-group ablation.**
  - A forward hook on `v_proj` zeros the output slice `[g·128 : (g+1)·128]` for KV group g.
  - This sets the attention output of both query heads in the group (heads 2g and 2g+1) to zero.
  - The hook runs before the KV-cache write, so cached values are also ablated.
  - Do not ablate through `k_proj`. `k_norm` follows it, and zeroing keys produces uniform attention rather than a removed component.
- **MLP ablation.** Either the whole block (a forward hook on `down_proj` zeroing its output) or selected intermediate neurons (a pre-hook on `down_proj` masking input channels; 3072 per layer).
- **Additional hook points.** `q_norm` and `k_norm` expose normalized queries and keys for measurement (Section 6.5) and for future interventions.
- **Ablation modes.**
  - *Zero ablation* replaces the component's output with zero.
  - *Mean ablation* replaces it with that component's mean output over a reference corpus (for example, 100k tokens of WikiText-2 train). Zero ablation moves activations outside the distribution the model was trained on; mean ablation is the more common choice in interpretability work.
  - Compute means separately for each precision, because quantization shifts them.
- **Required tests.**
  - An all-ones mask produces logits bitwise identical to running with no hook. Multiplying 16-bit values by 1.0 is exact.
  - Ablating query head h matches, within tolerance, a forward pass with columns `[h·128 : (h+1)·128]` of the `o_proj` weight set to zero.
  - Ablating KV group g matches, within tolerance, ablating query heads 2g and 2g+1 together.
  - A hook registered on a `QuantizedLinear` fires, and its observed output equals the module's return value.

### 5.6 Serving layer

- **Framework.** FastAPI with a single GPU worker that consumes an `asyncio` request queue. Only one generation runs at a time. Concurrent execution on one GPU would make tokens-per-second and latency telemetry meaningless.
- **Session state.** Each session stores an intervention configuration: precision, mask set, ablation mode, thinking mode, and sampling parameters. The configuration is stored as data, not as hooks registered on the model. The worker applies a session's configuration immediately before running that session's request and clears it afterward. The model object is shared between sessions, so hooks left attached to it would leak between users.
- **Model residency.**
  - The 16-bit, INT8, and INT4 copies are all resident.
  - The copies share one 16-bit embedding / LM-head tensor, plus the norm weights.
  - Switching precision changes which copy the worker uses and does not reload weights.
- **Prompt construction.**
  - The server builds prompts with the model's chat template.
  - Thinking mode uses the hard switch (`enable_thinking=True` or `False`), not the in-prompt `/think` and `/no_think` soft switches. The soft switches have no effect when thinking is disabled, and the hard switch is unambiguous.
  - Thinking content from earlier assistant turns is removed before those turns are sent back to the model, following the model card's guidance. The official template does this; the server must do the same if it assembles history itself.
- **Generation limits.** A per-request `max_new_tokens` cap, for example 512 in non-thinking mode and 2048 in thinking mode. Long thinking traces increase latency and KV-cache size (Appendix A).
- **Transport.** One WebSocket per session.
  - Server-to-client frames: generated tokens (tagged as thinking or answer content), the top-k next-token probabilities for each token, and telemetry samples.
  - Client-to-server frames: generation requests and configuration updates.

### 5.7 Frontend

The visual approach follows the Transformer Explainer (Georgia Tech Polo Club): an architecture diagram, attention heatmaps, a live next-token probability distribution, and temperature, top-k, and top-p controls. Its architecture does not transfer. It runs GPT-2 in the browser through ONNX Runtime, whereas this project's kernel runs on a server GPU. The frontend is a thin client over the API in Section 5.6. The framework is Svelte or React, chosen by the frontend owner, with D3 for visualizations.

| View | Content | Notes |
|---|---|---|
| Chat | Streaming generation, sampling controls, thinking-mode toggle | Thinking mode is off by default. Thinking content is shown in a separate collapsible region. Sampling defaults follow the model card: non-thinking temperature 0.7, top-p 0.8, top-k 20; thinking temperature 0.6, top-p 0.95, top-k 20 [1]. |
| Ablation panel | A 28 × 16 query-head grid, a 28 × 8 KV-group grid, MLP toggles, zero or mean mode | Grid dimensions come from model metadata. Selecting a KV group highlights the 2 query heads it feeds. |
| Precision toggle | 16-bit, INT8, INT4 | Applies to the current session. |
| Compare | Same prompt run under two configurations, side by side | Seeded sampling with identical parameters on both sides, not greedy decoding, following the model card. Given identical logits, seeded sampling is deterministic, so any difference comes from the configuration. |
| Next-token distribution | Top-k probabilities for each generated token | Taken from the logits already computed; costs nothing extra. |
| Attention heatmap | One selected layer and query head | Requires `attn_implementation="eager"` and `output_attentions=True`, because SDPA does not return attention weights. This returns weights for all 28 layers: 16 × T² × 2 bytes each, about 0.94 GB total at T = 1024. Limit heatmap requests to 1024 tokens and discard all layers except the selected one immediately. On demand only. |
| Telemetry | Decode tokens/s, TTFT, current context length, weight memory by precision, GPU clock, temperature, power | See Section 6.4 for how memory is reported. Context length is shown because it affects decode speed (Appendix A). |
| Experiment results | Component-importance heatmaps by precision and a rank-correlation scatter plot | Loaded from precomputed results (task 3.5). |

## 6. Measurement Methodology

### 6.1 Hardware
- **Primary device: NVIDIA GeForce RTX 4050 Laptop GPU.** Ada architecture (sm_89), 6 GB GDDR6, 96-bit bus, about 192 GB/s theoretical bandwidth, native FP16 and BF16, INT8 tensor cores. Every headline number in the final report comes from this device.
- **Rationale for a single device.** A performance comparison is valid only when both configurations run on the same system. Kernel tuning parameters (block size, rows per block, shared memory use, occupancy) are also architecture-specific.
- **Cross-architecture validation (Phase 4).** The same container image and autotuning sweep run on rented or university GPUs, for example an RTX 4090 (sm_89, same architecture, about 1 TB/s) and an A100 (sm_80). These results appear in a separate table. They are never mixed into primary-device comparisons.
- **University compute.** Request access in week 1, since approval can take weeks. Its use is the validation runs and, if available, the overnight experiment in task 3.5. It is not suitable for development: batch scheduling slows the edit-compile-profile loop, and compute nodes generally cannot host a public web server.

### 6.2 Laptop GPU variance control
Laptop GPUs change clocks with temperature and power budget. Every benchmark run must:
- Run on AC power with the maximum-performance power profile.
- Attempt clock locking with `nvidia-smi -lgc <freq>`. Laptop drivers may reject this; record whether it succeeded.
- Log SM clock, memory clock, temperature, and power at 100 ms intervals throughout the run, through NVML (`pynvml`) or `nvidia-smi --query-gpu=... -lms 100`.
- Discard runs in which the SM clock varies beyond a fixed threshold (for example, 5%).
- Run warmup iterations first, then report the median and interquartile range over N ≥ 50 iterations. Never report a single run.

### 6.3 Performance metrics
- **Timing.** Use `torch.cuda.Event` pairs with `torch.cuda.synchronize()`. Do not use wall-clock timing around asynchronous GPU calls, because it measures only enqueue time.
- **Kernel level.**
  - Time per call for each of the five (N, K) shapes in Section 4.1.
  - Effective bandwidth, calculated as bytes read and written (weights + scales + x + y) divided by time.
  - Effective bandwidth as a percentage of *measured* peak bandwidth. Measure the peak with a device-to-device copy or a streaming-read kernel on the same device; do not use the spec-sheet figure.
- **End to end.**
  - Time to first token (TTFT).
  - Decode tokens/s, excluding the first token.
  - Conditions for every configuration:
    - Non-thinking mode.
    - A fixed prompt, a fixed number of generated tokens with end-of-sequence ignored (so that repetition or early stopping does not change the token count), and seeded sampling with the non-thinking defaults.
  - Measure at context lengths of 512, 1024, and 4096 tokens. Because of GQA with head dim 128, KV-cache reads grow quickly with context and reduce the achievable speedup (Appendix A). Every reported number states its context length.
- **Profiling.**
  - Nsight Compute: per-kernel DRAM throughput, achieved occupancy, and memory-stall reasons.
  - Nsight Systems: the full decode timeline, including launch gaps and CPU-side overhead.
- **Expected discrepancy.** HF eager decoding launches several hundred kernels per token (28 layers × 7 linear layers alone is 196 GEMV launches), and Python and launch overhead may dominate end-to-end latency. Kernel-level speedup can therefore exceed end-to-end speedup. Report both, and use the Nsight Systems timeline to attribute the gap. S2 addresses launch overhead directly.

### 6.4 Memory metrics
- **Weight bytes by precision.** Computed analytically from buffer sizes (Appendix A) and confirmed with `torch.cuda.max_memory_allocated()` in a separate process for each precision.
- **Why separate processes.** The live server holds multiple copies of the model at once and shares the embedding tensor between them, so measuring inside it cannot attribute memory to one precision.
- **Headline figure.** Report allocated memory. State that `nvidia-smi` reports reserved memory plus CUDA context overhead, and that this figure does not drop when tensors are freed.

### 6.5 Numerical fidelity: the test ladder
Two sources of error are measured separately:
- **Quantization error.** The reference PyTorch backend against the 16-bit baseline measures the quality cost of the quantization scheme itself.
- **Kernel error.** The CUDA backend against the reference backend measures implementation correctness. Accumulation order differs between the two, so the comparison uses a tolerance rather than bitwise equality: relative error below 1e-3 against an FP32 reference.

| Rung | Check | What it detects |
|---|---|---|
| 0 | Baseline dtype check: FP16 vs. BF16 logits, and an inf/NaN and maximum-magnitude scan of every module output (task 1.5) | FP16 overflow on this model |
| 1 | Kernel unit tests over the five model shapes plus synthetic edge shapes (K not divisible by 512, N not divisible by rows per block), and a `compute-sanitizer` pass | Indexing errors, out-of-bounds access, race conditions |
| 2 | Per-layer activation cosine similarity and max absolute error, quantized vs. 16-bit, on fixed inputs. For attention, measure at the `q_proj` and `k_proj` outputs and again at the `q_norm` and `k_norm` outputs. | Which layer causes a divergence. Also measures how much projection error the Q/K norms absorb. |
| 3 | WikiText-2 test perplexity (fixed context length and stride), computed on raw text without the chat template | Quality loss too small to see in sampled text. Raw-text perplexity on a post-trained model is used as a relative measure across precisions, not as an absolute quality score. |
| 4 | Fixed-seed generation on a fixed set of prompts, in non-thinking mode | Qualitative failure; demo readiness |

### 6.6 Head-importance experiment (Section 2)
- **Components.** 448 query heads (28 × 16) and 224 KV groups (28 × 8), for each of 3 precisions and 2 ablation modes (zero and mean).
- **Metric.** ΔNLL_c = NLL(eval set, component c ablated) − NLL(eval set, clean), computed separately for each precision. Compare ΔNLL values across precisions, not raw NLL, because clean NLL differs between precisions.
- **Eval set.** Fixed held-out raw-text tokens, for example 32 sequences of 512 tokens from WikiText-2 test.
  - Measure the time for one full pass in task 1.16.
  - Set the eval-set size so that the whole sweep of 672 components × 3 precisions × 2 modes finishes within one overnight run, or two nights split by mode.
- **Model choice.** The experiment uses the served post-trained model, so the experiment and the demo describe the same model. `Qwen/Qwen3-0.6B-Base` exists and could be used in an ablation on the effect of post-training, but it is not part of the main experiment.
- **Noise floor.** Repeat the clean-model and one-ablation measurements on disjoint eval subsets to estimate ΔNLL variance. Report only precision-dependent differences that exceed this variance.
- **Analysis.**
  - Spearman ρ between precision pairs.
  - Jaccard overlap of the top-20 components.
  - A per-layer breakdown of importance shifts.
  - A comparison of importance shifts between query heads and KV groups.
  - Each analysis in both ablation modes.

## 7. Phased Plan

The owners are **K** (kernel and quantization track), **U** (interpretability and UI track), and **K+U** (joint).

### Phase 1: Foundation and vertical slice (Weeks 1–2)

Goal: a complete pipeline that runs correctly without custom CUDA. It loads a quantized model, generates through the real API, applies hooks, and reports fidelity metrics.

**Week 1**
- **1.1 [K] Development environment.** Native Ubuntu 24.04, NVIDIA driver, a CUDA toolkit version matching PyTorch's CUDA build, Nsight Compute, and Nsight Systems. Route display output to the integrated GPU. Enable profiling counters for non-root users (`NVreg_RestrictProfilingToAdminUsers=0`). Confirm with a trivial kernel profiled in Nsight Compute.
  *Rationale:* Kernel optimization depends on hardware counter access. WSL2 and GPUs that also drive a display both restrict profiling, and this should be found in week 1, not week 5.
- **1.2 [K] Container image.** A Dockerfile with pinned CUDA, PyTorch, Python, and `transformers` (≥ 4.51.0) versions, run through the NVIDIA Container Toolkit. Test it for one hour on a rented GPU by loading Qwen3-0.6B and generating text.
  *Rationale:* Older `transformers` versions cannot load the model. The Phase 4 cross-architecture runs and any recovery from laptop failure depend on an environment that can be reproduced in minutes.
- **1.3 [K+U] API contract and mock server.** Specify the WebSocket message schemas, including the thinking/answer token tag, and the REST endpoints (Section 5.6). Deploy a mock server that returns synthetic tokens, probabilities, and telemetry.
  *Rationale:* The API contract is where the two tracks connect. Fixing it on day one lets U build the entire UI in parallel and turns integration into a configuration change.
- **1.4 [K] Model inventory.**
  - Record `config.json` values and all `nn.Linear` module paths and shapes.
  - Confirm the tables in Section 4.1.
  - Confirm that `lm_head` and `embed_tokens` share one tensor.
  - Confirm that the `q_norm` and `k_norm` modules exist.
  *Rationale:* Kernel shapes, hook slice widths, and the ablation grid dimensions all come from this inventory.
- **1.5 [K] Baseline dtype decision.**
  - Run the model in BF16 and in FP16 on a text sample of at least 50k tokens, with hooks recording inf/NaN counts and the maximum absolute value at every module output.
  - Compare FP16 and BF16 logits and perplexity.
  - If FP16 shows any inf/NaN, or activations within a factor of 4 of 65,504, use BF16 as the baseline and activation dtype. Otherwise use FP16.
  - Record the decision and the data behind it.
  *Rationale:* The checkpoint is BF16. An FP16 baseline that overflows would corrupt every comparison in the project.
- **1.6 [K] Benchmark harness.** CUDA-event timing, warmup, median and IQR over N runs, concurrent clock, temperature, and power logging, run rejection, and CSV output. The harness takes context length and the number of generated tokens as parameters. Implement a measured-peak-bandwidth test.
  *Rationale:* Every later performance claim goes through this harness (Sections 6.2–6.3).
- **1.7 [K] 16-bit baselines.**
  - TTFT, decode tokens/s, and peak allocated memory, in non-thinking mode, at context lengths of 512, 1024, and 4096.
  - An Nsight Systems timeline of one decode step.
  *Rationale:* This is the comparison baseline, and it gives a first measurement of how much launch and Python overhead contributes to decode time.
- **1.8 [K] Request university compute access.**
  *Rationale:* Approval lead time.

**Week 2**
- **1.9 [K] Quantization library.** Per-channel INT8 quantize and dequantize functions working from the BF16 source weights. Unit tests on round-trip error distributions per layer.
- **1.10 [K] `QuantizedLinear` with the reference backend, plus model surgery.** Pure-PyTorch dequantize followed by `F.linear`. Surgery skips `lm_head` and verifies the shared tensor afterward (Section 5.4).
  *Rationale:* This gives a working quantized model before any CUDA exists, and it is the numerical reference the CUDA kernel is tested against.
- **1.11 [K] Fidelity ladder, rungs 2–4.** Per-layer activation comparison, including before and after the Q/K norms, WikiText-2 perplexity, and fixed-prompt generation outputs. Produce the first 16-bit vs. INT8-reference fidelity table.
- **1.12 [U] Hook prototypes.** Query-head, KV-group, and MLP ablation on the 16-bit model, with slice widths from `config.head_dim`. Implement the four tests in Section 5.5 and run them on the reference `QuantizedLinear` model as well.
- **1.13 [U] Chat UI against the mock server.** Streaming display, sampling controls, the thinking-mode toggle, and a separate display region for thinking content.
- **1.14 [K+U] Prompt builder.** Chat-template construction with the `enable_thinking` hard switch, and removal of thinking content from earlier turns in the history. Unit-tested against the official template's output.
- **1.15 [K+U] Vertical slice.** Replace the mock server's generation endpoint with the real model: 16-bit and INT8-reference, with ablation configurations applied per request.
- **1.16 [U] Experiment timing.** Measure one full ΔNLL pass. Set the eval-set size for task 3.5.

**Phase 1 exit criteria**
- The baseline dtype is chosen and the data behind the choice is recorded.
- The INT8 reference model generates through the real API in both thinking modes, and its perplexity is reported alongside the 16-bit baseline.
- Ablation hooks pass all four tests in Section 5.5 on both 16-bit and quantized models.
- 16-bit baselines are recorded with variance at all three context lengths.
- Nsight Compute counters are accessible on the primary device.

### Phase 2: Parallel tracks (Weeks 3–6)

**Kernel track [K]**
- **2.K1 (Week 3) Kernel v0 and the 16-bit control kernel.** Both templated on activation type. Build and bind through `cpp_extension`. Test against `F.linear` on all five model shapes.
  *Rationale:* Proves the build, binding, and test pipeline on the simplest kernel. The 16-bit control establishes the baseline that isolates the effect of precision.
- **2.K2 (Week 3) Kernel correctness suite.** Pytest parameterized over model shapes, synthetic edge shapes, and both activation dtypes, with FP32 reference comparison, tolerance checks, and `compute-sanitizer`. Runs on every kernel change.
- **2.K3 (Week 4) Kernel v1.** Warp per row, vectorized loads, FP32 accumulation, shuffle reduction, scale and optional bias epilogue. Profile DRAM throughput with Nsight Compute.
  *Rationale:* Coalesced vectorized loads are the main factor in approaching peak bandwidth.
- **2.K4 (Week 5) Kernel v2.** Shared-memory activation vector, multiple rows per block, K-tail handling (tested with synthetic shapes), and launch parameters as templates. Apply the same changes to the 16-bit control kernel.
- **2.K5 (Week 5) Autotuner.** Sweep block size and rows per block for each of the five (N, K) shapes. Store results as a per-device configuration file.
  *Rationale:* Moving to another device then requires rerunning the sweep, not rewriting the kernel (Section 6.1).
- **2.K6 (Week 6) Prefill path and backend dispatch.** Scratch-buffer dequantization followed by cuBLAS for T > 1, and the GEMV kernel for T = 1. Integrate into `QuantizedLinear` as `backend="cuda"`.
- **2.K7 (Week 6) Kernel benchmarks.** Per-shape time and percentage of measured peak bandwidth for INT8 v2, the 16-bit control, and cuBLAS.

**Interpretability and UI track [U]**
- **2.U1 (Week 3) Intervention engine.** The session configuration schema, and a hook manager that applies and removes a configuration around each request. Test for leakage: run two sessions with different configurations and confirm they are isolated.
- **2.U2 (Week 3) Mean-ablation statistics.** Compute and cache per-query-head, per-KV-group, and per-MLP mean activations for each precision.
- **2.U3 (Week 4) Server.** Request queue, single GPU worker, session store, prompt builder integration, generation limits, and the WebSocket protocol. Replaces the remaining mock endpoints.
- **2.U4 (Week 5) Ablation panel, precision toggle, and next-token distribution view.** The panel includes the query-head grid, the KV-group grid with head-to-group highlighting, and the MLP toggles.
- **2.U5 (Week 6) Telemetry panel, compare view, and attention heatmap.** The compare view uses seeded sampling. The heatmap enforces the 1024-token limit and keeps only the selected layer.

**Phase 2 exit criteria**
- The CUDA backend passes the full correctness suite for the chosen activation dtype.
- Kernel v2 per-shape bandwidth efficiency is reported alongside the 16-bit control and cuBLAS.
- All UI views work against the real server using the reference backend.

### Phase 3: Integration and experiment (Weeks 7–8)

- **3.1 [K] Serve the CUDA backend.** Switch the served quantized model from `backend="reference"` to `"cuda"`.
- **3.2 [K+U] Hook verification on CUDA modules.** Run the Section 5.5 test suite on the CUDA-backend model.
  *Rationale:* The custom op returns a new tensor from inside `forward`. The tests confirm that hooks still intercept the tensor that downstream layers actually consume.
- **3.3 [K] Full fidelity ladder on the CUDA backend.** Confirm that CUDA-backend perplexity equals reference-backend perplexity within noise, and that kernel error stays within tolerance across all layers.
- **3.4 [K] End-to-end benchmarks.** TTFT and decode tokens/s for 16-bit, INT8-reference, and INT8-CUDA at all three context lengths. An Nsight Systems timeline comparison attributing the gap between kernel-level and end-to-end speedup (Section 6.3). Compare measured speedup against the context-dependent ceilings in Appendix A.
- **3.5 [U] Head-importance experiment.** Run the Section 6.6 sweep for 16-bit and INT8 in zero and mean modes. Export results for the experiment-results view.
- **3.6 [K+U] System tests.** Concurrent sessions queued correctly, no leakage of configuration between sessions, precision switching with no weight reload, thinking-mode switching, generation limits enforced, and a one-hour soak test for memory growth.

**Phase 3 exit criteria**
- The demo serves the INT8 CUDA backend.
- The 16-bit vs. INT8 fidelity, performance, and component-importance results are complete.

### Phase 4: Extensions, validation, documentation (Weeks 9–10)

- **4.1 [K] INT4.** Group-wise quantization, packing, a GEMV kernel variant with per-group scaling, and correctness tests.
- **4.2 [K+U] INT4 results.** Fidelity ladder, performance, and the component-importance experiment at INT4.
  *Rationale:* Round-to-nearest INT4 on a 0.6B-parameter Qwen3 model is expected to degrade measurably [2]. Three precision points give a curve of quality and importance shift against bits per weight, and the degradation is itself a result. If quality is too low for interactive use, the demo exposes INT4 only in the experiment-results view.
- **4.3 [K] Cross-architecture runs.** Container image plus autotune sweep on rented or university GPUs. Report a separate table of per-device bandwidth efficiency.
- **4.4 [K+U] TinyLlama replication (optional).**
  - Apply model surgery to TinyLlama-1.1B-Chat (head dim 64, 32 / 4 GQA, no Q/K normalization, separate embedding and LM head).
  - Rerun the INT8 fidelity ladder and the Section 6.6 experiment.
  *Rationale:* Tests whether the quantization-induced importance shift holds on an architecture without Q/K normalization and with a different GQA ratio. TinyLlama also gives a second data point for the bandwidth ceiling, which is higher (1.88×) because its LM head is smaller.
- **4.5 [K+U] Documentation.** Kernel design and memory-traffic analysis (Appendix A, expanded), API reference, reproduction instructions (container image, benchmark commands, experiment scripts), and methodology.
- **4.6 [K+U] Final report and demo recording.**

### Stretch items (in priority order)
- **S1. Fused INT8 prefill GEMM** using INT8 tensor cores (`mma.sync` or CUTLASS), replacing scratch-buffer dequantization.
- **S2. CUDA graphs for decode.** Ablation is implemented as persistent mask tensors that are always multiplied in, and a configuration change writes new mask values instead of registering hooks. The captured graph stays valid when the configuration changes. Removes the launch overhead identified in task 3.4.
- **S3. Fused QKV and gate/up projections.** Fewer kernel launches. This changes the module structure: hooks on individual `k_proj` or `v_proj` modules stop working, so KV-group ablation must move to a hook point placed after the fused output is split. The fused output must be split before `q_norm` and `k_norm`. Implement only after the Phase 3 results are final.
- **S4. Quantized LM head.**
  - The shared 16-bit embedding / LM-head tensor is 41% of the INT8 model's per-token weight traffic (Appendix A).
  - Quantizing it requires an INT8 copy used only as the LM head, while the embedding lookup continues to use the 16-bit tensor.
  - Cost: 0.16 GB of additional memory.
  - Measure the effect on perplexity and on decode speed.

## 8. Ownership and Interfaces

| Interface | Owner | Consumer | Fixed by |
|---|---|---|---|
| WebSocket and REST schema | K+U | Both | Week 1 (task 1.3) |
| Telemetry fields | K | U (dashboard) | Week 1 (task 1.3) |
| Baseline activation dtype | K | Both | Week 1 (task 1.5) |
| `QuantizedLinear` module paths and names | K | U (hook targets) | Week 2 (task 1.10) |
| Prompt builder behavior (thinking mode, history) | K+U | Both | Week 2 (task 1.14) |
| Session configuration schema | U | K (worker applies it) | Week 3 (task 2.U1) |
| Experiment result format | U | Both (report, UI) | Week 7 |

Changing any interface after the date listed requires agreement from both owners.

## 9. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| INT8 kernel does not beat the 16-bit baseline | No speedup result | The 16-bit control kernel isolates the precision effect. Report percentage of measured peak bandwidth regardless. The reference backend keeps the demo working. |
| End-to-end speedup hidden by Python and launch overhead | Modest tokens/s gain | Report kernel-level and end-to-end numbers separately; attribute the gap with Nsight Systems; S2. |
| Speedup ceiling limited by the large-vocabulary LM head and KV cache | Headline speedup at most 1.59×, and lower at long context | Report measured speedup against the context-dependent ceilings in Appendix A; S4; optional TinyLlama comparison (task 4.4). |
| FP16 overflow on a BF16-trained model | Corrupted baseline and comparisons | Task 1.5 dtype check; kernel templated on activation dtype. |
| Head slices computed from `hidden_size // num_heads` | Ablations hit the wrong channels | Slice widths always read from `config.head_dim`; test in Section 5.5. |
| Model surgery breaks the embedding / LM-head sharing | Doubled memory or a mismatched LM head | Surgery skips `lm_head`; pointer-equality check after surgery. |
| Thinking mode produces long outputs | Slow demo, large KV cache, lower measured speedup | Non-thinking default; generation limits; benchmarks in non-thinking mode at fixed context lengths. |
| Greedy decoding causes repetition | Poor comparison-view output | Seeded sampling with model-card parameters. |
| INT4 quality collapse | Unusable INT4 demo | Report the degradation as a result; INT4 limited to the experiment view if necessary; demo defaults to INT8. |
| Thermal and clock variance on the laptop | Irreproducible numbers | Section 6.2 protocol; clock logging; run rejection. |
| Profiling counters unavailable | Cannot measure kernel efficiency | Checked in task 1.1; native Linux with display on the integrated GPU. |
| Late integration failure | Project does not work end to end | Vertical slice in week 2; mock server in week 1; interfaces fixed early (Section 8). |
| Configuration leaking between sessions | Wrong results in the demo | Configuration stored as data, applied per request; leakage test in task 2.U1. |
| Experiment runtime too long | Incomplete results | Eval-set size set from measured pass time (task 1.16); split across nights by ablation mode; run on rented GPU if necessary. |

## 10. Deliverables

1. Source repository: CUDA kernels, PyTorch extension, quantization library, intervention engine, server, frontend.
2. Container image and reproduction scripts.
3. Correctness test suite and fidelity results for all precisions, including the baseline dtype check.
4. Performance results: kernel-level bandwidth efficiency and end-to-end metrics at three context lengths on the primary device, plus the cross-architecture table.
5. Component-importance experiment results and analysis (query heads and KV groups, both ablation modes, all precisions).
6. The deployed interactive application and a demo recording.
7. A technical report covering design, methodology, results, and limitations.
8. Optional: TinyLlama replication results.

## 11. Open Decisions

- **Course evaluation emphasis.** If the course grades a working system, the vertical slice and demo polish come first. If it grades a research result, Section 6.6 takes priority over UI polish. Confirm with the instructor in week 1.
- **Default ablation mode shown in the UI.** Zero ablation is easier to explain to viewers; mean ablation is the more defensible choice for interpretation.
- **Thinking mode in the demo.** Off by default. Decide whether the UI exposes it to all users, and whether ablation effects on thinking traces are shown as a demo feature.
- **TinyLlama replication (task 4.4).** Include it only if Phases 1–3 finish on schedule.
- **Frontend framework.** Svelte or React, decided by U.

---

## Appendix A: Memory Traffic and Bandwidth Ceilings (Qwen3-0.6B)

Per decode step, each weight of every linear layer and of the LM head is read once. The ceilings below assume 192 GB/s and count only weight and KV-cache traffic. They exclude activations, non-GEMV kernels, and launch overhead, so they are upper bounds. Measured throughput will be lower, and the ratio of measured throughput to these ceilings is the efficiency metric. MB = 10⁶ bytes.

**Parameter counts**

| Component | Count |
|---|---|
| Attention linears per layer | q 1024×2048 + k 1024×1024 + v 1024×1024 + o 2048×1024 = 6.29M |
| MLP linears per layer | 3 × 1024 × 3072 = 9.44M |
| All quantized linears | 28 × 15.73M = 440.4M |
| Shared embedding / LM head (16-bit) | 151,936 × 1024 = 155.6M |

**Weight bytes read per decoded token, excluding the KV cache**

| Precision | Linears | LM head | Total | Ceiling | Speedup ceiling |
|---|---|---|---|---|---|
| 16-bit | 880.8 MB | 311.2 MB | 1192.0 MB | 161 tok/s | — |
| INT8 | 440.4 MB | 311.2 MB | 751.6 MB | 255 tok/s | 1.59× |
| INT4, g = 128 | 220.2 + 6.9 MB (scales) | 311.2 MB | 538.3 MB | 357 tok/s | 2.21× |

INT4 scale overhead assumes FP16 group scales (2 bytes per 128 weights). INT8 per-channel FP32 scales (1.4 MB) are omitted.

The LM head accounts for 26% of 16-bit traffic and 41% of INT8 traffic. This is the motivation for S4.

**KV-cache traffic**

Each token of context adds 28 layers × 2 (K and V) × 8 KV heads × 128 × 2 bytes = 114.7 KB, all of which is read at every decode step. KV-cache reads are the same at every weight precision, so they lower the speedup ceiling as context grows.

| Context length | KV bytes per step | KV as % of INT8 weight traffic | INT8 speedup ceiling | INT4 speedup ceiling |
|---|---|---|---|---|
| 0 (weights only) | 0 | 0% | 1.59× | 2.21× |
| 512 | 58.7 MB | 8% | 1.54× | 2.10× |
| 1024 | 117.4 MB | 16% | 1.51× | 2.00× |
| 4096 | 469.8 MB | 63% | 1.36× | 1.65× |

Qwen3-0.6B stores more KV cache per token than Qwen1.5-0.5B (98.3 KB) or TinyLlama-1.1B (22.5 KB). Its 8 KV heads of dimension 128 give it half of Qwen1.5's KV heads but twice the head dimension, and it has 4 more layers. A single thinking-mode response can reach several thousand tokens. This is the reason benchmarks use non-thinking mode and report context length.

**VRAM budget on the RTX 4050 Laptop (6 GB)**

| Component | Size |
|---|---|
| 16-bit model (linears + shared embedding / LM head) | 0.88 + 0.31 = 1.19 GB |
| INT8 linears + scales (embedding shared) | 0.44 GB |
| INT4 linears + scales (embedding shared) | 0.23 GB |
| KV cache (1 sequence, 4096 tokens, 16-bit) | 0.47 GB |
| Prefill scratch buffer | 0.01 GB |
| Attention heatmap at the 1024-token limit (all layers, before discarding) | 0.94 GB peak |
| **Total** | **about 3.3 GB peak** |

About 2.7 GB of headroom remains for activations, the CUDA context, mean-ablation statistics, and allocator overhead. Loading TinyLlama at the same time for task 4.4 (2.2 GB at 16-bit) would raise the peak to about 5.5 GB and leave almost no headroom. Run task 4.4 in a separate process with the Qwen3 server stopped.

**Reference: TinyLlama-1.1B-Chat (task 4.4)**

| Precision | Bytes per token (weights only) | Ceiling | Speedup ceiling |
|---|---|---|---|
| 16-bit | 1937.8 + 131.1 = 2068.8 MB | 93 tok/s | — |
| INT8 | 968.9 + 131.1 = 1100.0 MB | 175 tok/s | 1.88× |
| INT4, g = 128 | 484.4 + 15.1 + 131.1 = 630.6 MB | 304 tok/s | 3.28× |

---

## References

1. Qwen Team. *Qwen3-0.6B model card and config.json.* Hugging Face, `Qwen/Qwen3-0.6B`. Source for the architecture values in Section 4.1, the thinking-mode behavior, and the recommended sampling parameters.
2. Zheng, X. et al. *An Empirical Study of Qwen3 Quantization.* arXiv:2505.02214, 2025. Evaluates RTN, GPTQ, AWQ, SmoothQuant, and BiLLM on Qwen3 at 1–8 bits. Reports larger low-bit degradation than for LLaMA3.
3. Cho, A. et al. *Transformer Explainer.* Georgia Tech Polo Club, `poloclub.github.io/transformer-explainer`. Reference for the frontend's visual design.
