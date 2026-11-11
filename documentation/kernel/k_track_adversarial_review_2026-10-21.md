# K-track adversarial review — 2026-10-21

> Historical record. Statements below describe the recorded source state.
> See the [documentation index](../README.md) for current context and related work.


## Review identity and scope

- **Review ID:** K-REVIEW-001.
- **Perspective:** principal engineer and research scientist reviewing numerical correctness, experimental validity, CUDA integration, and delivery risk.
- **Code baseline:** HEAD `91e4de8` (`Initial model quantization`), plus the working-tree implementation inspected during the preceding review. Much of the benchmark harness was uncommitted. The commit alone does not identify the reviewed source state.
- **Documentation:** [K-track guide](k_track_guide.md), [findings](k_track_findings.md), and [project plan revision 3](pre_trained_fast_lab_plan_v3.md).
- **Implementation inspected:** the quantization library, fidelity evaluator, benchmark package, experiment scripts, CUDA bandwidth extension, tests, backend, package configuration, and notebook code.
- **Execution evidence:** the preceding review ran `quant/.venv/bin/python -m pytest -q quant/tests`: **51 passed, 1 skipped, 15 deselected**. It also ran targeted CPU reproductions described below.
- **GPU limitation:** `nvidia-smi` could not communicate with the NVIDIA driver in this session. Reported GPU measurements were examined as repository artifacts; they were not independently reproduced.
- **Change scope:** this document records the review. It does not implement fixes. At the user's request, no staging, commits, or edits to other files accompany it.

This is a dated snapshot, not a continuously rewritten verdict. Preserve its findings and evidence when later work supersedes them. Use the stable identifiers below to connect fixes and subsequent reviews to the original concerns.

## Executive assessment

The K track has a strong numerical and experimental foundation. The distinction between quantization error, implementation error, and performance is explicit and mostly reflected in the code. The model inventory, local-versus-compounded fidelity measurements, position-zero analysis, and cache-aware microbenchmarks are particularly good work.

The central systems claim remains unproven: there is no custom inference GEMV implementation yet. `backend="cuda"` raises `NotImplementedError`; the existing CUDA source is a bandwidth-measurement kernel. The completed work establishes a quantized reference model and measurement infrastructure, rather than an accelerated inference engine.

The highest-priority weaknesses are a comparator that silently loses linear-layer coverage when the reference model is quantized, an unresolved numerical mismatch between the planned prefill path and reference, permissive benchmark reporting after rejected measurements, and incomplete experiment provenance. These issues are tractable, but they should be resolved before performance and fidelity claims become acceptance gates for optimized kernels.

There is no evidence here that the reported default FP16-versus-INT8 perplexity result is invalid. There is also insufficient evidence to promote the projected end-to-end speedup to a measured expectation or the eight-prompt smoke test to a general quality claim.

## Findings index

Priority describes recommended sequencing, not a claim that every issue is a present production failure. **P1** means address before relying on the next major validation milestone; **P2** means address before reproducible research or deployment claims; **P3** means a bounded follow-up or defensive improvement. All findings are open at this review snapshot.

| ID | Priority | Finding | Evidence class |
|---|---|---|---|
| K-001 | P1 | Quantized references silently lose linear activation coverage | Reproduced on CPU |
| K-002 | P1 | Planned prefill and reference have different numerical semantics | Code/design analysis |
| K-003 | P1 | Rejected or incomplete benchmarks can still produce successful reports | Code inspection |
| K-004 | P2 | Sliding-window scoring guarantee fails without overlap | Reproduced on CPU |
| K-005 | P2 | Result provenance cannot reconstruct the exact experiment | Code and repository inspection |
| K-006 | P2 | Queue-ahead timing needs independent validation as workloads change | Measurement-method concern |
| K-007 | P2 | Bandwidth reference and estimated decode breakdown are overstated in places | Artifact/design interpretation |
| K-008 | P2 | Numerical reference and future kernel tolerance contracts need definition | Code/design analysis |
| K-009 | P2 | Source-relative runtime resources need packaging support | Package-layout inspection |
| K-010 | P2 | Backend is not yet an instrument for controlled comparisons | Code inspection |
| K-011 | P2 | Importance experiments need paired effects and uncertainty | Research-design concern |
| K-012 | P3 | Q/K norm error measurements do not establish a mechanism | Research interpretation |
| K-013 | P3 | Lifecycle and input contracts need defensive coverage | Code inspection |

## Strengths to preserve

### A meaningful separation of concerns

The project distinguishes weight reconstruction error, local activation error, error accumulated through the network, corpus-level fidelity, and runtime. These answer different questions. Retain that structure when adding CUDA and INT4 rather than collapsing acceptance to a single perplexity number.

The reference computes the planned per-channel expression with scale multiplication after the dot product. A matching FP16 control kernel is a strong planned experiment: comparison against that control helps separate narrower weight traffic from implementation quality. cuBLAS remains the practical external baseline.

### Architecture-specific facts are verified

The inventory explicitly checks head dimensions, GQA, the seven decoder linear layers, Q/K norms, and embedding/LM-head sharing. Using the real `head_dim` instead of deriving it from hidden size avoids a particularly serious ablation bug on this model.

Restricting quantization to decoder linears gives the first implementation a clear scope. Keeping the tied LM head at 16-bit also exposes a real limit on total compression and speedup; the documentation acknowledges that limit.

### Fidelity investigation is unusually careful

Local comparisons on common inputs help distinguish an individual layer's approximation from upstream drift. Reporting position zero separately avoids letting a large residual activation dominate the aggregate. The layer-27 cancellation analysis is useful evidence that relative error alone can mislead.

The BF16 model is a useful yardstick for the scale of numerical changes. It should remain contextual evidence rather than an equivalence threshold.

### Benchmarking addresses real failure modes

Rotating weight copies addresses L2 residency. Batched timing addresses event granularity. A separate NVML process avoids GIL competition. Random buffer contents reduce one source of unrealistic memory measurements. Recording rejected attempts is good practice even though their downstream treatment needs improvement.

These are worth preserving. The next step is validating the remaining assumptions, not replacing the harness wholesale.

## Detailed findings

### K-001 — Quantized references silently lose activation coverage

**Location:** `quant/src/quant/fidelity.py`, `hook_points()` and `compare_activations()`.

Hook discovery identifies linear modules using `isinstance(module, nn.Linear)`. Local comparison selection uses the same restriction. `QuantizedLinear` is an `nn.Module`, not an `nn.Linear`.

**Reproduction:** construct a tiny Qwen3, quantize its decoder, deep-copy it, and compare the two quantized models. In a one-layer model the compounded points were only:

```text
model.layers.0.self_attn.q_norm
model.layers.0.self_attn.k_norm
model.layers.0
```

The local point count was zero. All seven linear points were absent.

**Impact:** current FP16-reference comparisons retain their linear points. The future INT8-reference-versus-CUDA comparison would silently omit the very modules it is intended to validate. A passing aggregate report could therefore conceal missing coverage.

**Recommended action:** recognize both linear implementations or derive targets from an explicit, checked module-path inventory. Reject mismatched inventories instead of silently producing a smaller report. Avoid dependence on the reference implementation's concrete class where possible.

**Closure evidence:** compare two quantized models and assert all seven linears, both Q/K norms, and the residual per layer are represented; verify expected local points; deliberately perturb one quantized linear and show the relevant statistics detect it. Recheck hook behavior if local comparisons invoke modules that also carry instrumentation hooks.

### K-002 — Prefill and reference implement different arithmetic

**Locations:** `quant/src/quant/quantized_linear.py`, `forward()`; plan §5.3; guide §9.4.

The reference evaluates:

```text
cast_output(scale * FP32_dot(FP32(x), FP32(q)))
```

The proposed prefill evaluates a different expression:

```text
F.linear(x, round_to_FP16(q * scale))
```

Rounding each reconstructed weight before multiplication is an extra approximation. It is distinct from reduction-order differences and final-output rounding. FP32 accumulation in the GEMM would not remove that distinction.

**Impact:** CUDA-versus-reference discrepancies will include an intentional prefill approximation. Prefill also produces the KV cache, so its differences affect subsequent decode even if the GEMV implementation is excellent. Fixed-sequence perplexity predominantly tests multi-token execution; it cannot establish decode-kernel correctness by itself.

**Recommended action:** document arithmetic separately for reference, prefill, and decode. Establish tolerances empirically rather than assuming the guide's proposed perplexity difference below 0.01% is attainable. Validate each path independently and then the transition between them.

**Closure evidence:** shared-input prefill output errors, teacher-forced cached decode errors, cache-versus-full-forward comparisons within each backend, and full corpus fidelity. Explain any accepted difference against INT8-reference. A correct result may justify revising the target rather than forcing the implementation to match a mathematically different reference.

### K-003 — Rejection does not consistently gate valid results

**Locations:** `quant/scripts/bench_linear.py`, `peak_bandwidth.py`, and `bench_e2e.py`.

The linear script summarizes the final timing attempt even when it was rejected. Its size-matched read uses the final attempt's times without carrying that attempt's acceptance status into the reported ratio. The peak-bandwidth script likewise builds headline values from final attempts. The end-to-end script records incomplete accepted-run counts and token mismatches but still returns success.

**Impact:** metadata may disclose a problem while the headline table and process exit status still invite use of the result. A valid numerator divided by an invalid bandwidth reference is not a valid efficiency estimate.

**Recommended action:** distinguish diagnostic output from publishable results. Keep rejected data, but exclude it from valid summaries. Mark the whole comparison invalid when either measurement fails acceptance. Return failure when the requested run count, determinism requirement, or other mandatory condition is not met.

This does not show that the stored accepted measurements are wrong. It is a failure-path weakness in the harness.

**Closure evidence:** exercise exhausted retries, rejected matched reads, incomplete end-to-end runs, and deliberately different tokens; confirm invalid measurements cannot become successful headline results.

### K-004 — Sliding-window scoring contract has an overlap precondition

**Location:** `quant/src/quant/fidelity.py`, `sliding_window_eval()`.

The function promises every token after the first is scored once. With nonoverlapping windows, each later window lacks the preceding token needed to score its first token.

**Reproduction:** 24 tokens, context 8, stride 8 yielded 21 scored tokens rather than 23. Stride 7 yielded 23. The default context 1024 and stride 512 use overlap and are not implicated by this reproduction.

**Recommended action:** require `ctx >= 2`, `0 < stride < ctx`, and at least two evaluation tokens if preserving the current promise. Alternatively document and implement different scoring semantics explicitly. Reject empty model collections and invalid reference names clearly.

**Closure evidence:** a hand-checked multiwindow likelihood test covering both full and partial final windows, plus invalid-parameter tests. The current one-window manual comparison is insufficient to validate window boundaries.

### K-005 — Provenance does not identify the exact experiment

**Locations:** `quant/src/quant/loading.py`, `data.py`, `bench/telemetry.py`, experiment output schemas, and Git state.

Model and tokenizer loading do not pin immutable revisions. Dataset loading does not pin a revision. Benchmark metadata records HEAD but not the dirty source state. Much of the reviewed harness was outside HEAD. Earlier fidelity and weight-error artifacts do not all carry the benchmark environment schema, despite the guide's broad statement that every result file records run conditions.

**Impact:** two results labeled with the same model ID and Git commit need not represent the same experiment. A lockfile helps package reproducibility but does not fix mutable model/data revisions or uncommitted code.

**Recommended action:** record model/tokenizer revisions, dataset identity and revision, a token-stream hash, exact arguments, precision policy, software/toolchain versions, device identity, and source-state identity. A dirty flag is useful but alone cannot reconstruct the patch; retain a source snapshot or artifact identifier for formal runs. Link downstream bandwidth comparisons to the exact bandwidth artifact and verify device compatibility.

**Closure evidence:** start from a clean checkout and the recorded inputs, regenerate a selected result, and explain the allowed hardware-dependent variation. Ensure every headline artifact points to a reconstructable source state.

### K-006 — Queue-ahead timing is a workload-dependent assumption

**Location:** `quant/src/quant/bench/timing.py`.

The harness queues a sleep before enqueueing measurements to prevent GPU starvation by Python. This is sensible, but a fixed delay does not prove the queue remains populated throughout an arbitrary workload. Enqueue cost changes with backend, allocation behavior, iteration count, instrumentation, and extension dispatch.

The delay uses a fixed cycle conversion. It is not a guaranteed 20 ms wall-clock delay across SM clocks; the guide's observed clocks exceed 2 GHz. This is a limitation of the explanatory claim, not evidence that existing samples are contaminated.

**Recommended action:** validate representative timing loops with Nsight Systems and sweep the queue-ahead delay. Compare against a pre-enqueued or graph-based microbenchmark where practical. Use preallocated outputs for low-level kernel timing when allocation is outside the intended operation contract, and report wrapper-inclusive latency separately if useful.

**Closure evidence:** no starvation gaps in the representative timed region and stable medians across reasonable queue-ahead settings. Repeat this check when the optimized kernel becomes much faster than the current operations.

### K-007 — Empirical references and estimates need precise labels

**Locations:** guide §5.7 and §9; findings §7; benchmark reports.

The size-matched read is an empirical bandwidth reference. It is not a hard performance ceiling: stored results already show cuBLAS exceeding it. GEMV has a different instruction mix, launch configuration, and memory-access behavior.

Likewise, the approximately 59% linear share of decode and projected 1.24× end-to-end improvement are estimates assembled from isolated timings and an estimated LM-head cost. They are not an observed trace breakdown. Real decode also includes a changing cache length, other kernels, allocation behavior, and CPU scheduling.

**Recommended action:** reserve “upper bound” for explicitly derived bounds with stated assumptions. Call the size-matched number a bandwidth reference or target. Label assembled timing budgets as forecasts. Measure the LM head and actual execution breakdown before treating the inferred residual time as launch overhead.

Additional portability limits: telemetry assumes NVML device zero; `environment()` also queries device zero even when CUDA visibility may remap devices. The theoretical-bandwidth calculation explicitly assumes GDDR6 and should not be reused unchanged for HBM devices. A memory-clock stability statistic is recorded, but acceptance currently checks only SM-clock spread.

**Closure evidence:** trace-based timing attribution at short and long context, consistent same-device denominator selection, and an explicit validity policy for telemetry. Report speedups across independently repeated runs or sessions; many samples from one thermal trajectory do not provide the same evidence as independent sessions.

### K-008 — Reference precision and tolerances need explicit contracts

**Locations:** `quantized_linear.py`, fidelity scripts, future GEMV correctness suite.

Calling `.float()` specifies storage dtype but does not by itself freeze every backend's internal multiplication precision policy. Set and record the FP32 matmul policy for oracle runs. PyTorch documents TF32 and other precision controls in its [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda) and [numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

Do not use one relative threshold for all outputs. Cancellation and near-zero references can make elementwise relative error arbitrarily large. Conversely, aggregate relative L2 can hide isolated bad outputs. Position-zero separation is useful, but kernel bugs should not receive a blanket exemption at difficult positions.

**Recommended action:** combine aggregate relative L2, maximum absolute error, and elementwise absolute-plus-relative checks. Use a small FP64 oracle suite for independence. Cover realistic and adversarial inputs: alternating signs, cancellation, zeros, large and small scales, bias, and synthetic tails.

**Closure evidence:** justify tolerances against arithmetic and dtype rather than a single observed run. Distinguish local implementation error from model-level accumulation. Validate cached decode directly, since multi-token perplexity primarily exercises prefill dispatch.

### K-009 — Runtime resources depend on the source checkout

**Locations:** `quant/src/quant/data.py`, `ext.py`, `pyproject.toml`.

`prompts.json` and `csrc/` are found by walking from source files into the repository layout. Package discovery alone does not establish that those resources will be included and found in a wheel. The scripts and output paths similarly assume a writable source tree.

**Recommended action:** explicitly package required resources or define a source-only deployment contract. Separate runtime resource lookup from experiment output directories. Build and test the intended distribution before containerization relies on it.

**Closure evidence:** install the built artifact into a clean environment away from the checkout and exercise prompt loading and CUDA source discovery. No wheel build was performed in this review; this is an identified packaging gap, not an observed wheel-install failure.

### K-010 — Backend cannot yet support controlled experiments

**Locations:** `backend/api.py`, `generation.py`, `interventions.py`, `model.py`, and schemas.

The endpoint uses generation settings but ignores intervention configuration, returns hardcoded latency values, and does not select quantized precision. There is no single-worker request queue or session-scoped hook lifecycle. The model loads at module import, and loading does not explicitly choose the K-track baseline dtype.

The intervention functions use appropriate hook locations for head and MLP masking, but their existence does not establish endpoint integration or leakage protection. `run_generation()` is unfinished and refers to a prompt on an experiment configuration that does not define one.

**Recommended action:** keep K-track measurement scripts independent while building a minimal real vertical slice. Establish a common model-loading and configuration contract. Add hook-equivalence, all-ones identity, cache behavior, and exception-safe cleanup checks before calling the service an experimental instrument.

The notebooks are exploratory evidence. Some comparisons vary thinking mode or repetition penalty, and at least some cells leave hook cleanup incomplete. Their outputs should not support causal ablation claims without controlled reruns.

**Closure evidence:** an API request demonstrably applies precision and interventions, reports measured latency, and leaves the next request unaffected. The service should serialize GPU work and have a clear seeded comparison policy.

### K-011 — Importance shifts require paired effects and uncertainty

**Location:** plan §2 and §6.6; future experiment implementation.

The shared research question gives the two tracks a coherent purpose. However, rank correlation and top-k overlap alone can overstate changes among nearly tied or noisy component scores.

For component c, define the ablation effect within precision p:

```text
effect(p, c) = NLL(p, ablated c) - NLL(p, intact)
interaction(c) = effect(quantized, c) - effect(FP16, c)
```

The interaction directly asks how quantization changes the measured ablation effect. Preserve per-block paired measurements so uncertainty can be estimated over meaningful corpus units. Tokens are dependent; naive independent-token uncertainty estimates are not sufficient.

**Recommended action:** define the evaluation set and selection rules before exploring component rankings. Use development data for tuning and a held-out set for final claims. Report uncertainty in effects and rank summaries, document top-k selection, and treat a null INT8 result as informative. Many-component searches require care when highlighting apparent outliers.

Precision-specific mean ablations alter both the model and its replacement value. That is a valid question, but it differs from holding the replacement fixed. Consider a shared-mean sensitivity analysis and label the two estimands clearly.

**Closure evidence:** a written protocol, paired block outputs, reproducible component masks, uncertainty estimates, and robust conclusions across reasonable ablation definitions. Avoid claiming architecture-wide generality from one checkpoint and corpus.

### K-012 — Q/K norm findings are observations, not explanations

**Locations:** `quant/scripts/fidelity.py`, absorption summaries; findings §6.

The reported local after/before relative-error ratios—about 1.15 for Q and 0.81 for K—are useful measurements. They do not by themselves identify a mechanism. Relative L2 before and after a learned normalization mixes direction error, radial error, channel weighting, and changes in reference magnitude. Averaging ratios across layers also differs from taking a ratio of aggregate errors.

**Recommended action:** decompose projection error into radial and directional parts per head. Compare normalization before and after learned gain. Examine attention-score or attention-distribution changes to connect representation error with behavior. Identify the low-cosine tokens instead of interpreting only minima and means.

The large activation at position zero and final-layer cancellation are documented observations. They justify targeted tests, not a relaxed correctness standard. The claim that the final residual output at position zero only affects its own logits is consistent with the final layer's ordering; it should not be generalized to earlier-layer position-zero perturbations, whose keys and values influence later tokens.

**Closure evidence:** a bounded mechanism study with controlled inputs and per-head measurements, or explicitly retaining the result as descriptive evidence only.

### K-013 — Defensive contracts and lifecycle behavior need coverage

**Locations:** `quantize.py`, `quantized_linear.py`, and `surgery.py`.

The main path handles real finite BF16 checkpoint weights well. Several edge contracts remain implicit:

- Nonfinite inputs are not explicitly rejected by the quantizer. NaN or infinity can invalidate scales and integer conversion without a clear error.
- Surgery preflights FP16 weights, but other failures can still occur after replacements begin. The comment about refusal leaving the model untouched is narrower than a general transactional guarantee.
- Existing hook handles attached to old linear objects are not transferred by replacement. Preserving paths supports registration after surgery; it does not preserve preexisting registrations.
- `_apply()` protects scales during dtype casts by hiding them and inferring a destination device from an empty tensor. This is practical for ordinary `.to()` calls, but does not establish compatibility with every `_apply` lifecycle operation, meta initialization, or storage-management API.
- `QuantizedLinear` is a useful inference replacement, but lacks the `.weight` interface some generic model tooling expects. “Drop-in” should be scoped accordingly.

**Recommended action:** state the supported inference lifecycle and validate invalid inputs at clear boundaries. Add targeted checks when new deployment paths require them rather than attempting universal framework compatibility now.

The refusal to quantize FP16 is a conservative provenance policy. It should not be described as proof that every BF16-to-FP16 conversion necessarily corrupts normal-range weights: BF16 has fewer significand bits, and many values convert exactly. The real risks include range and subnormal behavior, plus losing assurance about the input's origin. Quantizing from the original checkpoint remains the right reproducibility rule.

**Closure evidence:** explicit supported contracts, clear failures for invalid weights, and coverage for the lifecycle operations actually used in deployment.

## Interpretation of existing numerical results

The stored WikiText-2 result, FP16 perplexity 19.8865 versus INT8-reference 19.9489, supports the claim that per-channel RTN INT8 closely preserves corpus likelihood under this protocol. It does not establish general task equivalence. The absolute perplexity increase is about 0.0624, the relative increase about 0.313%, and the corresponding mean ΔNLL about 0.00313 nats per scored token.

The eight generation prompts are smoke tests. “No correct response became incorrect” is a qualitative observation about those outputs. The same seed makes a comparison easier to inspect, but sampled continuations diverge rapidly after the first changed token and do not isolate local probability changes.

The BF16 yardstick is helpful, but “17 times the BF16 perplexity change” is sensitive to a very small denominator. Lead with ΔNLL, relative perplexity, KL, and agreement instead. Likewise, zero clipping under max-based scaling is largely a consequence of the quantizer construction; it is not independent evidence of task quality.

FP32 scales are a sound choice for the INT8 reference and initial kernel. The observed subnormal FP16 scales support caution, but subnormal does not automatically mean unusable or zero. For INT4, measure scale rounding and underflow directly. Group scales use a different denominator and smaller groups, so their range cannot be inferred mechanically from INT8 row scales.

The reported reduction from roughly 1.110 GiB to 0.730 GiB allocated after loading is a whole-model reduction of about 34%, or a size ratio of about 1.52×. Describe compression at both decoder-weight and whole-model levels. An INT8 decoder does not imply an entire model half the size when the shared vocabulary tensor remains 16-bit.

## Kernel-development recommendations

1. **Integrate correctness before optimizing.** Implement v0 and its FP16 control, bind them, validate them, and connect v0 to cached generation immediately. Keep the reference multi-token fallback until the prefill contract is settled. Avoid waiting until v3 to discover model-integration failures.
2. **Treat shared-memory staging as a measured hypothesis.** A 2–6 KB activation vector can already receive cache reuse. Staging adds cooperative loads, synchronization, and resource costs. Retain it only when profiling and timings support it.
3. **Test alignment separately from tails.** Allocation alignment does not ensure every row is aligned when synthetic K changes. A final-iteration tail branch cannot by itself make arbitrary row-start `uint4` accesses safe.
4. **Define the extension boundary.** Check dtype, shape, device agreement, contiguity/strides, alignment, bias layout, and launch constraints. Preserve current-stream execution and device guards. The bandwidth extension is a useful existing pattern.
5. **Validate both real and hostile inputs.** Include all five model shapes, N and K tails, nondefault streams, supported devices, cancellation, scales, and bias. Run memory sanitization; add race checking when shared memory appears.
6. **Measure the operation users execute.** Distinguish low-level preallocated kernel time, Python-wrapper time, and model decode time. Keep cache state and traffic accounting explicit. The INT8-reference “effective bandwidth” is a logical-byte normalization, not measured physical DRAM traffic, because it materializes FP32 weights and intermediates.
7. **Keep autotuning interpretable.** Save device identity, source version, configuration, validity, and timing variation. Do not select a winner from noise and present its best sample as a reproducible improvement.

The intended claim is reduced memory traffic for single-sequence decode. Avoid expanding prematurely into activation quantization or tensor-core prefill. In particular, an “INT8 tensor-core GEMM” for W8A16 requires an explicit treatment of operand conversion or activation quantization; the plan's stretch wording does not define a free direct replacement for the current arithmetic.

## Recommended sequence and acceptance gates

| Order | Work | Evidence required before moving on |
|---|---|---|
| 1 | Repair validation and reporting contracts | Quantized-reference coverage; rejection paths; valid evaluator parameters; source/input provenance |
| 2 | Complete FP16 baselines and profiling | Accepted runs at all contexts; actual LM-head/GEMV/other/idle breakdown; repeated-run conditions |
| 3 | Implement and integrate v0 plus FP16 control | Local correctness, sanitizer results, cached generation, stream/device checks |
| 4 | Optimize v1/v2 and tune | Measured improvement with uncertainty and profiling explanation; no correctness regression |
| 5 | Implement explicit prefill semantics | Separate prefill/decode budgets; cache-transition tests; full fidelity ladder |
| 6 | Produce end-to-end CUDA results | Valid matched runs, measured speedup, memory results, trace-based attribution |
| 7 | Extend to INT4 and importance experiments | Packing/scale correctness, quality evaluation, paired-effect protocol |

Backend work can proceed alongside this sequence, but the research scripts should remain usable without the service. The K track should not be blocked from validating its first kernel by unfinished UI features.

## Protocol for subsequent adversarial reviews

Use a new dated review with a new review ID. Reference these finding IDs instead of silently rewriting this document. A finding closes when the stated failure is addressed and the relevant evidence exists, not merely when code changes or a test passes.

For each subsequent review, record:

- exact source state, including uncommitted changes;
- which claims changed and the artifacts supporting them;
- tests independently run versus results read from artifacts;
- hardware and access limitations;
- findings closed, still open, superseded, or newly introduced;
- remaining assumptions that could reverse the conclusion.

Distinguish a demonstrated defect from a design risk, a research hypothesis, or a portability limitation. Challenge favorable interpretations as carefully as unfavorable ones. Do not reject a useful implementation because it misses an unjustified numerical target, and do not excuse a correctness gap because the generated text looks coherent.

## Reproduction notes and boundaries

The CPU suite command used in the preceding review was:

```bash
quant/.venv/bin/python -m pytest -q quant/tests
```

The comparator reproduction used a one-layer randomly initialized BF16 Qwen3 with tied embeddings, quantized through `quantize_model_()`, then compared against a deep copy through `compare_activations()`. The sliding-window reproduction used 24 random vocabulary tokens with context 8 and strides 7 and 8. Neither reproduction required model downloads or a GPU.

No inference CUDA kernel, sanitizer run, wheel-install test, container build, or GPU performance reproduction was performed by this review. The execution results establish the reported CPU behaviors only. Existing GPU numbers remain repository-reported measurements pending independent rerun.

The documentation directory is currently ignored by Git. This file has intentionally not been staged or committed, and no ignore rules were changed. The findings index above provides document-level indexing for future reviews; repository tracking remains for the user to perform after inspection.
