# First CUDA Kernel Phase: Correctness and Cached Decode

- **ID:** K-PLAN-001
- **Status:** Draft — awaiting agreement; no kernel implementation authorized by this document
- **Created:** 2026-10-25
- **Updated:** 2026-10-25
- **Scope:** Quant; the first working INT8 CUDA linear backend and its validation gates
- **Code baseline:** `16b56072363df674f8e1c4cdf9d972e799fb5820` on `KernelDevelopment`
- **Source context:** Planning inspection initially observed documentation/ignore-rule
  changes, subsequently committed at the baseline above. This document and the
  documentation reorganization are later, uncommitted documentation changes.
- **Related:** [Documentation index](../README.md),
  [project plan revision 3](../kernel/pre_trained_fast_lab_plan_v3.md),
  [K-track guide](../kernel/k_track_guide.md),
  [findings](../kernel/k_track_findings.md),
  [K-REVIEW-001](../kernel/k_track_adversarial_review_2026-10-21.md)

## 1. Goal and review scope

This draft defines the first working INT8 CUDA linear backend and its validation
gates. It deliberately ends before vectorization, shared-memory optimization,
autotuning, or performance claims. Proposed contracts and thresholds below are
for review, not evidence that they have been implemented or validated.

**Success means:** Qwen3 can prefill through the existing reference arithmetic,
then execute single-token decode through a tested custom CUDA GEMV, with module
paths and hooks preserved.

The user selected **correctness plus v0** as the first milestone, including the
matching 16-bit control and cached generation with reference prefill. Approval
of that milestone's direction is not approval of every detail in this draft.

The ultimate goal remains serving quantized Qwen through API endpoints using the
custom CUDA linear backend. This milestone establishes the working model path
needed before optimization and service integration.

## 2. Documentation claims mapped to the implementation

The planning review covered all seven Markdown documents present in
`documentation/` before this reorganization, the homepage mockup, relevant code
and tests, and locally available result artifacts. Historical documents remain
dated records; this plan does not rewrite their original findings. Previously
removed PDF progress reports were not part of the available documentation reviewed.

| Document claim or insight | Concrete implementation or evidence | Planning consequence |
| --- | --- | --- |
| Plan v3 section 4.1 and guide section 5.1: seven decoder linears per layer, five unique shapes, tied embedding/LM head | [Model inventory](../../quant/scripts/inventory.py) asserts the architecture; local `quant/results/inventory.json` reports 196 decoder linears, 440,401,920 decoder parameters, and no inventory failures | Test all five shapes; leave `lm_head` and embedding sharing unchanged |
| Plan section 5.1 and findings section 4: symmetric per-row INT8, FP32 scales, BF16 source weights | [`quantize_int8_per_channel`](../../quant/src/quant/quantize.py) and [quantizer tests](../../quant/tests/test_quantize.py) | Reuse the format and quantizer without changing the quantization scheme |
| Guide section 5.2 and findings section 5: FP16 is the selected activation baseline | [`BASELINE_DTYPE`](../../quant/src/quant/loading.py), the decision rule in [dtype checking](../../quant/scripts/dtype_check.py), and the stored `float16` decision | FP16 is primary; instantiate and test BF16 too |
| Guide section 5.3: some real scales are below FP16's normal range | [QuantizedLinear](../../quant/src/quant/quantized_linear.py) registers FP32 scales and protects them in `_apply()` | Read and apply scales in FP32; include small-scale regressions |
| Guide section 5.4: reference evaluates the dot product before applying the row scale | `QuantizedLinear.forward()` computes `F.linear(x.float(), qweight.float()) * scales`, adds bias, then casts | Match this ordering; do not round reconstructed weights to FP16 first |
| Review K-001: quantized references lose linear activation coverage | [`hook_points()` and `compare_activations()`](../../quant/src/quant/fidelity.py) discover only `nn.Linear`; CPU reproduction yielded three compounded points and zero local points for a one-layer INT8 comparison | Repair the comparator before using it to validate CUDA |
| Review K-002: planned FP16 prefill differs numerically from the reference | Plan section 5.3 proposes rounding `q * scale` into 16-bit weights; no such path exists yet | Keep reference arithmetic for multi-token execution in this milestone |
| Review K-004: scoring every token once requires overlap | `sliding_window_eval()` lacks parameter validation; CPU reproduction with 24 tokens and context/stride 8 scored 21 rather than 23 tokens | Add overlap and input checks before relying on corpus evaluation |
| Findings section 6: INT8-reference perplexity is about 19.9489 versus FP16 19.8865 | Local `quant/results/fidelity_int8_ref.json` matches; [fidelity script](../../quant/scripts/fidelity.py) produces these measurements | Historical quality evidence, not a universal threshold or proof of decode correctness |
| Guide section 5.7: 183.4 GB/s read rate and context-512 FP16 decode around 86.7 tok/s | Local bandwidth and decode artifacts match; decode artifacts contain five accepted smoke runs | Do not treat them as 50-run, all-context baselines or first-phase performance gates |
| Review K-003/K-006/K-007: reporting and timing assumptions need tightening | [Linear benchmark](../../quant/scripts/bench_linear.py) summarizes the final attempt even when rejected; matched-read validity is not propagated; [timing](../../quant/src/quant/bench/timing.py) uses a fixed queue-ahead delay | Resolve before publishing speedup claims; do not use these reports as correctness gates |
| Review K-009: runtime resources depended on the checkout | Current [data loading](../../quant/src/quant/data.py), [extension loading](../../quant/src/quant/ext.py), [package configuration](../../quant/pyproject.toml), and [resource validation](../integration/quant-resource-validation.md) document the repair | Add sources under packaged `csrc` and retain resource lookup |
| Review K-010: backend intervention handling was incomplete | Current [API](../../backend/src/kernel_backend/api.py) calls [`run_generation()`](../../backend/src/kernel_backend/generation.py), which applies interventions and removes hooks in `finally`; model loading and latency placeholders still need further work | Some historical criticism is superseded; quantized API serving remains a later task |
| Migration documents: one environment and passing baseline suites | [Independent baselines](../integration/independent-baselines.md), [workspace validation](../integration/workspace-validation.md), and [resource validation](../integration/quant-resource-validation.md) | Preserve dependency versions and rerun existing suites alongside new tests |
| Homepage mockup: precision selection and comparisons | [UI design reference](../mockups/Homepage.png); current API has no quantized model selection | Preserve the eventual API goal without adding UI/API work to this milestone |

The existing [bandwidth CUDA implementation](../../quant/src/quant/csrc/bandwidth.cu)
measures streaming reads. It is useful infrastructure and a stream/device-handling
example, but it is not an inference kernel.

During planning, `/usr/local/cuda-13.0/bin/nvcc --version` reported **13.0.88**.
The earlier shell's `nvcc not found` log reflected PATH visibility. No inference
kernel was compiled and no new GPU performance measurements were made during
planning. The quoted numerical/performance artifacts were inspected, not rerun.

## 3. Kernel and integration contract

### Arithmetic and interfaces

Add lazy Python wrappers backed by a JIT extension:

- `int8_gemv(x, qweight, scales, bias=None)`
- `gemv_16bit_control(x, weight, bias=None)`

The INT8 operation computes, per output row:

```text
FP32 dot product of x and signed INT8 weights
→ FP32 row-scale multiplication
→ optional FP32 bias addition
→ cast to the activation dtype
```

The 16-bit control uses the same thread mapping and accumulation structure, with
16-bit weights and no quantization scale. It is a structural comparison kernel,
with its own numerical reference.

| Property | Proposed first-phase contract |
| --- | --- |
| Activations | Contiguous CUDA `[1, K]`, FP16 or BF16 |
| INT8 weights | Contiguous row-major `[N, K]`, signed INT8 |
| Scales | Contiguous CUDA `[N]`, FP32 |
| Optional low-level bias | Contiguous CUDA `[N]`, FP32 |
| Control weights | Contiguous `[N, K]`, same dtype as activations |
| Output | Newly allocated `[1, N]`, activation dtype and device |
| Dimensions | Positive N and K; arbitrary scalar-loop tails |
| Execution | Calling device's current CUDA stream, guarded device context |
| Supported use | Eager inference; reject gradient-requiring execution when gradients are enabled |

`QuantizedLinear` retains its existing bias storage format and passes an FP32
view/conversion to the low-level operation. No state-dictionary keys change.

Implement **v0 as one thread per output row**, with 128 threads per block, scalar
loads, FP32 accumulation, and bounds checks. Do not introduce vector loads, shared
memory, atomics, fast-math compilation, or tuning in this phase.

Validate metadata at the extension boundary before allocation or launch: ranks,
dimensions, dtype, contiguity, device agreement, and bias layout. Numerical
correctness is defined for finite inputs and representable outputs; avoid
synchronizing tensor-value scans on every decode call.

### Module dispatch

Preserve `backend="reference"` as the default. Implement the existing
`backend="cuda"` choice as follows:

- Preserve input shape `[..., K]` and return `[..., N]`.
- Flatten leading dimensions to determine the number of input vectors.
- One vector: normalize activation contiguity and invoke v0.
- More than one vector: execute the existing FP32 reference expression.
- Empty leading dimensions: return the correctly shaped empty output without a
  CUDA launch.
- Reject unsupported dtype/device combinations clearly; do not silently route an
  invalid CUDA request to CPU execution.
- Load the extension only when the single-vector CUDA path is exercised.

This dispatch counts vectors, not merely sequence length. A two-sequence,
one-token batch uses the reference fallback in this milestone.

Preserve load order: **BF16 checkpoint → quantization/surgery → activation dtype
cast**. Preserve decoder module paths and embedding/LM-head sharing.

## 4. Implementation sequence and gates

### Commit 1 — Repair the relevant validation contracts

- Recognize both `nn.Linear` and `QuantizedLinear` in activation discovery and
  local comparisons.
- Require matching decoder inventories and feature dimensions; fail on missing
  or mismatched targets.
- Add INT8-versus-INT8 identity coverage: ten compounded and nine local points
  per Qwen layer.
- Perturb one quantized linear deliberately and prove local statistics detect it.
- Verify comparator-installed hooks are removed after failures.
- Require `ctx >= 2`, `0 < stride < ctx`, at least two effective tokens, a nonempty
  model collection, and a valid reference name for sliding-window evaluation.
- Add a hand-checked overlapping multiwindow likelihood test, including a partial
  final window.

**Gate:** the two CPU reproductions above are covered by regression tests, and
existing CPU tests pass.

### Commit 2 — Add v0 and the matching 16-bit control

- Place the CUDA implementation under packaged `quant/src/quant/csrc/`.
- Extend the existing resource-aware JIT loader with a separate GEMV extension.
- Add lazy wrappers with the contracts above.
- Use FP32 accumulation and a device guard/current stream; retain architecture
  8.9 as the existing default, respecting an explicit architecture override.
- Add the numerical and boundary tests below.
- Warm-build the extension, then run its focused tests through Compute Sanitizer
  memcheck with a nonzero error exit code.

**Gate:** both kernels pass correctness tests and memcheck on the primary GPU.
Missing hardware or tools is incomplete validation, not a passing gate. NVIDIA
documents memcheck for out-of-bounds and misaligned accesses; racecheck becomes
relevant when shared memory is introduced.
[Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

### Commit 3 — Integrate CUDA decode with reference prefill

- Extract the reference expression into a reusable internal method without
  changing its arithmetic.
- Implement dispatch and replace the existing CUDA-not-implemented test with
  supported-path and invalid-input tests.
- Verify forward and pre-forward hooks still observe or modify the tensors
  actually consumed.
- Verify unchanged buffers, scales, module paths, and state-dictionary round trips.
- Instrument tests to prove one-vector execution invokes the real CUDA operation
  and multi-vector execution does not.

**Gate:** tiny-Qwen prefill followed by cached decode works for both activation
dtypes, with explicit custom-kernel call coverage.

### Commit 4 — Validate the real model and installed distribution

- Construct reference and CUDA models from identical quantized buffers.
- Add optional `revision` forwarding to model/tokenizer loading so validation can
  use one immutable checkpoint revision.
- Use the eight packaged prompts in non-thinking mode.
- Derive a fixed 16-token continuation from the reference, then feed those same
  tokens to both models with independent caches.
- Compare cached logits step by step, and compare cached execution against
  full-prefix execution within each backend.
- Record single-vector calls across all 196 decoder linears; distinguish cached
  token positions from the original prompt's position zero.
- Include a short seeded generation smoke check. Matching sampled text is
  diagnostic rather than the numerical correctness oracle.
- Build and install a wheel outside the checkout, then exercise actual GEMV
  execution from that installation using the shared dependency environment.
- Record validation results and update current usage documentation without
  rewriting historical reports.

**Gate:** all existing suites and new tests pass, wheel execution works, and the
evidence identifies the tested source and checkpoint.

## 5. Test policy, evidence, and deferred work

### Numerical tests

Test all five `(N, K)` model shapes:

```text
(2048, 1024), (1024, 1024), (1024, 2048),
(3072, 1024), (1024, 3072)
```

Also test small and awkward shapes, including K values 1, 31, 33, 511, 513, and
3073, with N values that leave partial blocks. Include contiguous views with
storage offsets so tests do not assume 16-byte alignment.

Use seeds 0, 1, and 42, both activation dtypes, and bias/no-bias cases. Include
zeros, signed extrema, alternating signs, cancellation, small scales spanning
the observed real range, and large finite activations.

Use three complementary checks:

1. FP32 reference arithmetic with IEEE FP32 matmul explicitly selected and restored
   after tests.
2. An independent FP64 oracle for small/adversarial cases.
3. Per-element error bounds plus aggregate relative L2 and maximum absolute error
   reporting.

For the FP64 check, calculate a conservative FP32 forward-error envelope using
`γ(2K+4)`, where `γ(m) = m·2⁻²⁴ / (1 − m·2⁻²⁴)`, applied to the sum of absolute
scaled products and bias. Round both envelope endpoints to the output dtype and
require the kernel output to lie within them. Exact-zero and bias-only cases also
receive exact expected-output tests.

For well-conditioned regression inputs, additionally require relative L2 below
`1e-3` for FP16 and `1e-2` for BF16. Cancellation cases use the absolute envelope
rather than an unstable relative ratio. No position receives a blanket correctness
exemption.

For cached/full-prefix model comparisons, use proposed smoke-test gates of:

- FP16: relative L2 ≤ `5e-3`, with `rtol=5e-3`, `atol=5e-2`.
- BF16: relative L2 ≤ `2e-2`, with `rtol=2e-2`, `atol=2e-1`.

These are proposed engineering gates, not measured results. Investigate failures;
do not automatically loosen tolerances. Reduction-order differences preclude a
general bitwise-equality requirement.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
[FP32 precision controls](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)

Also test invalid metadata, nondefault streams, device-context restoration, input
immutability, and mixed-device rejection when multiple GPUs are available.

### Evidence to retain

Record source commit or reconstructable source snapshot, lockfile hash, toolchain
versions, GPU/driver, architecture setting, precision policy, seeds,
checkpoint/tokenizer revision, prompt/token hashes, test logs, sanitizer output,
and wheel identity.

Separate independently executed checks from historical artifacts. Keep generated
logs outside Git and summarize them in documentation.

### Explicitly deferred

- Warp-per-row/vectorized v1, shared-memory experiments, autotuning, INT4, graphs,
  and projection fusion.
- FP16 scratch-buffer prefill and its separate numerical budget.
- Published speedup claims and the incomplete all-context performance baselines.
- Benchmark rejection/provenance repairs beyond first-phase correctness evidence.
- API model selection, request serialization, measured endpoint latency, and UI
  changes.

K-001 and K-004 are first-phase repair targets. K-002 is avoided by retaining
reference prefill. K-008 receives a kernel-specific precision contract. K-003,
K-005–K-007, and remaining lifecycle/research concerns stay explicitly open for
their appropriate milestones. Partial work on provenance does not close K-005.

After this phase, optimization can proceed against a working model path. Serving
quantized Qwen through the API with the custom CUDA backend remains the end goal.

## 6. Decision log

| Date | Decision or open question | Rationale / evidence |
| --- | --- | --- |
| 2026-10-25 | User selected correctness plus v0 as the first milestone | Establish working cached decode before optimizing or publishing performance claims |
| 2026-10-25 | Draft proposes reference multi-vector fallback | Avoids the prefill/reference arithmetic mismatch in K-002 |
| 2026-10-25 | Detailed contracts, numerical gates, and commit boundaries await review | Saving this plan does not authorize implementation |

## 7. Completion evidence

No implementation commits, new inference-kernel tests, sanitizer results, or
installed GEMV execution results exist for this plan yet. Add links here only
after the work is implemented and independently validated.
