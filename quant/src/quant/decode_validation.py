"""Acceptance gates for CUDA cached decode (K-PLAN-002 stages 3 and 4).

Three comparisons run for every prompt and continuation row:
    G1  CUDA cached       vs reference cached       kernel integration in cached decode
    G2  reference cached  vs reference full-prefix  the reference model's own cache-path rounding noise
    G3  CUDA cached       vs CUDA full-prefix       cached CUDA decode against uncached execution

The model-level gates are relative to the measured noise (decisions of 2026-10-28; the never-measured absolute
tolerances of K-PLAN-001 sit inside this model's 16-bit noise, so G2 failed them with no kernel involved):
    G2  finite, max relative L2 <= gate.noise_bound, top-1 agreement >= gate.min_top1. A broken cache path
        or teacher-forcing helper produces O(1) errors and cannot pass.
    G1, G3  finite, top-1 agreement >= gate.min_top1, max relative L2 <= gate.noise_bound, and each statistic
        in gate.statistics (median, p90) at most max(gate.noise_factor x G2's, gate.local_rel_l2). The max is
        held to the absolute bound, not to G2's max: a single tail row of two combined noise sources is not
        evidence of a defect, while an O(1) error still fails.
Statistics pool every row of every prompt. Two kernel-level checks accompany them:
    coverage  only around the CUDA cached pass, every QuantizedLinear path makes exactly tokens - 1
              single-vector kernel calls, and no other pass makes any.
    local     every kernel call in that pass is compared with the reference expression on the same input
              (FP32, before the cast): finite and relative L2 <= gate.local_rel_l2, as in stage 2.
"""

import hashlib
import math
import statistics
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from quant import gemv
from quant.decode import cached_decode_logits, full_prefix_logits, greedy_continuation
from quant.precision import ieee_fp32_matmul
from quant.quantized_linear import QuantizedLinear


@dataclass(frozen=True)
class Gate:
    noise_factor: float  # G1 and G3 statistics may be at most this multiple of G2's
    noise_bound: float  # absolute bound on the max relative L2 of G1, G2 and G3
    min_top1: float  # top-1 agreement required of G1, G2 and G3
    local_rel_l2: float  # per kernel call; also the floor of the relative limits when G2 is near zero
    statistics: tuple[str, ...] = ("median", "p90")  # held relative to G2


GATES = {
    torch.float16: Gate(noise_factor=1.5, noise_bound=2e-2, min_top1=0.98, local_rel_l2=1e-3),
    torch.bfloat16: Gate(noise_factor=1.5, noise_bound=1e-1, min_top1=0.98, local_rel_l2=1e-2),
}
COMPARISONS = {
    "G1": ("cuda_cached", "ref_cached"),
    "G2": ("ref_cached", "ref_full"),
    "G3": ("cuda_cached", "cuda_full"),
}
CHECKS = (*COMPARISONS, "coverage", "local")
UNKNOWN = "<unknown>"


#||test - ref|| / ||ref|| in FP64. Zero reference: 0 if test is also zero, else infinite.
def rel_l2(test: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (test.double() - ref.double()).norm().item()
    norm = ref.double().norm().item()
    if norm == 0:
        return 0.0 if diff == 0 else math.inf
    return diff / norm


#Per-row metrics for test [L, V] against ref [L, V]. A row with any non-finite value has infinite error.
def compare_rows(test: torch.Tensor, ref: torch.Tensor) -> list[dict]:
    if test.shape != ref.shape:
        raise ValueError(f"shapes differ: {tuple(test.shape)} vs {tuple(ref.shape)}")
    rows = []
    for i in range(ref.shape[0]):
        t, r = test[i].float(), ref[i].float()
        finite = bool(torch.isfinite(t).all()) and bool(torch.isfinite(r).all())
        rows.append({
            "row": i,
            "finite": finite,
            "rel_l2": rel_l2(t, r) if finite else math.inf,
            "max_abs_err": (t.double() - r.double()).abs().max().item() if finite else math.inf,
            "top1_agree": finite and bool(t.argmax() == r.argmax()),
        })
    return rows


#Median, 90th percentile (linear interpolation) and max of the rows' relative L2, and top-1 agreement.
def summarize_rows(rows: list[dict]) -> dict:
    values = [r["rel_l2"] for r in rows]
    finite = all(r["finite"] for r in rows) and all(math.isfinite(v) for v in values)
    summary = {
        "rows": len(rows),
        "all_finite": finite,
        "top1_agreement": sum(r["top1_agree"] for r in rows) / len(rows),
        "max_abs_err": max(r["max_abs_err"] for r in rows),
    }
    if not finite:
        return {**summary, "median": math.inf, "p90": math.inf, "max": math.inf}
    p90 = statistics.quantiles(values, n=10, method="inclusive")[8] if len(values) > 1 else values[0]
    return {**summary, "median": statistics.median(values), "p90": p90, "max": max(values)}


#Verdicts for G1-G3 from pooled summaries; also returns the limits G1 and G3 were held to.
def evaluate_gates(summaries: dict, gate: Gate) -> tuple[dict, dict]:
    noise = summaries["G2"]
    checks = {
        "G2": noise["all_finite"] and noise["max"] <= gate.noise_bound and noise["top1_agreement"] >= gate.min_top1
    }
    limits = {stat: max(gate.noise_factor * noise[stat], gate.local_rel_l2) for stat in gate.statistics}
    for gid in ("G1", "G3"):
        s = summaries[gid]
        checks[gid] = (
            s["all_finite"]
            and s["top1_agreement"] >= gate.min_top1
            and s["max"] <= gate.noise_bound
            and all(s[stat] <= limits[stat] for stat in gate.statistics)
        )
    return checks, {**limits, "max": gate.noise_bound}


def _qweight_paths(model) -> dict[int, str]:
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, QuantizedLinear)]
    paths = {m.qweight.data_ptr(): name for name, m in modules}
    if len(paths) != len(modules):
        raise RuntimeError("QuantizedLinear modules share qweight storage; calls cannot be attributed")
    return paths


@contextmanager
def _wrapped_kernel(wrapper):
    original = gemv.int8_gemv
    gemv.int8_gemv = wrapper(original)
    try:
        yield
    finally:
        gemv.int8_gemv = original


@contextmanager
def kernel_call_counter(model):
    """Counts int8_gemv calls per QuantizedLinear path of `model`, attributing each call by the identity of
    the qweight it receives. Calls with any other weight count as UNKNOWN. Restores the kernel on exit."""
    paths = _qweight_paths(model)
    counts = Counter()

    def wrapper(kernel):
        def recording(x, qweight, scales, bias=None):
            counts[paths.get(qweight.data_ptr(), UNKNOWN)] += 1
            return kernel(x, qweight, scales, bias)
        return recording

    with _wrapped_kernel(wrapper):
        yield counts


@contextmanager
def kernel_local_errors(model):
    """Compares every int8_gemv output with the reference expression on the same input (FP32, before the
    cast). Yields {path: {"calls", "max_rel_l2", "all_finite"}}. Restores the kernel on exit."""
    paths = _qweight_paths(model)
    errors = {}

    def wrapper(kernel):
        def checking(x, qweight, scales, bias=None):
            y = kernel(x, qweight, scales, bias)
            ref = F.linear(x.float(), qweight.float()) * scales
            if bias is not None:
                ref = ref + bias
            finite = bool(torch.isfinite(y).all())
            entry = errors.setdefault(paths.get(qweight.data_ptr(), UNKNOWN),
                                      {"calls": 0, "max_rel_l2": 0.0, "all_finite": True})
            entry["calls"] += 1
            entry["all_finite"] = entry["all_finite"] and finite
            entry["max_rel_l2"] = max(entry["max_rel_l2"], rel_l2(y, ref) if finite else math.inf)
            return y
        return checking

    with _wrapped_kernel(wrapper):
        yield errors


#Exact coverage: the observed path set equals `expected_paths`, each with `calls_per_path` calls, none unknown.
def coverage_check(counts: Counter, expected_paths: list[str], calls_per_path: int) -> dict:
    observed = {p: n for p, n in counts.items() if p != UNKNOWN}
    expected = dict.fromkeys(expected_paths, calls_per_path)
    return {
        "passed": observed == expected and counts.get(UNKNOWN, 0) == 0,
        "expected_paths": len(expected),
        "observed_paths": len(observed),
        "calls_per_path": calls_per_path,
        "expected_total": calls_per_path * len(expected),
        "observed_total": sum(counts.values()),
        "unknown_calls": counts.get(UNKNOWN, 0),
        "missing": sorted(set(expected) - set(observed)),
        "extra": sorted(set(observed) - set(expected)),
        "wrong_counts": {p: n for p, n in sorted(observed.items()) if p in expected and n != calls_per_path},
    }


def local_check(errors: dict, expected_paths: list[str], gate: Gate) -> dict:
    worst = max((e["max_rel_l2"] for e in errors.values()), default=math.inf)
    return {
        "passed": set(errors) == set(expected_paths)
        and all(e["all_finite"] and e["max_rel_l2"] <= gate.local_rel_l2 for e in errors.values()),
        "paths": len(errors),
        "calls": sum(e["calls"] for e in errors.values()),
        "max_rel_l2": worst,
        "limit": gate.local_rel_l2,
        "worst_path": max(errors, key=lambda p: errors[p]["max_rel_l2"]) if errors else None,
    }


def _token_hash(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.to(torch.int64).cpu().numpy().tobytes()).hexdigest()


def run_decode_validation(
    model_ref, model_cuda, prompts: list[torch.Tensor], tokens: int, gate: Gate, *, overrides: dict | None = None
) -> dict:
    """Runs the four passes per prompt under the IEEE FP32 policy and evaluates G1-G3, coverage and the
    local kernel check.

    prompts: 1-D int64 prompt tensors. tokens: continuation length L (>= 2, so decode steps always run),
    derived greedily from model_ref. overrides: {pass name: fn(model, prompt, continuation)} replacing a
    pass, used by failure-path tests. Returns a JSON-serializable report with an overall "passed".
    """
    if tokens < 2:
        raise ValueError(f"tokens must be at least 2 so that decode steps run, got {tokens}")
    if not prompts:
        raise ValueError("prompts must not be empty")
    passes = {
        "ref_cached": (model_ref, cached_decode_logits),
        "ref_full": (model_ref, full_prefix_logits),
        "cuda_full": (model_cuda, full_prefix_logits),
        "cuda_cached": (model_cuda, cached_decode_logits),
    }
    overrides = overrides or {}
    expected_paths = [name for name, m in model_cuda.named_modules() if isinstance(m, QuantizedLinear)]
    results = []
    with ieee_fp32_matmul() as policy:
        for prompt in prompts:
            continuation = greedy_continuation(model_ref, prompt, tokens)
            logits, coverage, local = {}, {}, None
            for name, (model, fn) in passes.items():
                checker = kernel_local_errors(model_cuda) if name == "cuda_cached" else nullcontext({})
                with kernel_call_counter(model_cuda) as counts, checker as errors:
                    logits[name] = overrides.get(name, fn)(model, prompt, continuation)
                if name == "cuda_cached":
                    coverage[name] = coverage_check(counts, expected_paths, tokens - 1)
                    local = local_check(errors, expected_paths, gate)
                else:
                    total = sum(counts.values())
                    coverage[name] = {"passed": total == 0, "observed_total": total}
            comparisons = {gid: compare_rows(logits[a], logits[b]) for gid, (a, b) in COMPARISONS.items()}
            results.append({
                "prompt_tokens": prompt.numel(),
                "prompt_sha256": _token_hash(prompt),
                "continuation": continuation.tolist(),
                "summaries": {gid: summarize_rows(rows) for gid, rows in comparisons.items()},
                "comparisons": comparisons,
                "coverage": coverage,
                "local": local,
                # Diagnostic: prefill runs the reference arithmetic on both models.
                "prefill_row_bitwise_identical": bool(torch.equal(logits["cuda_cached"][0], logits["ref_cached"][0])),
            })
    pooled = {gid: summarize_rows([row for r in results for row in r["comparisons"][gid]]) for gid in COMPARISONS}
    checks, limits = evaluate_gates(pooled, gate)
    checks["coverage"] = all(c["passed"] for r in results for c in r["coverage"].values())
    checks["local"] = all(r["local"]["passed"] for r in results)
    return {
        "passed": all(checks[key] for key in CHECKS),
        "checks": {key: checks[key] for key in CHECKS},
        "gate": asdict(gate),
        "pooled": pooled,
        "relative_limits": limits,
        "tokens": tokens,
        "quantized_linear_paths": len(expected_paths),
        "matmul_policy": policy,
        "prompts": results,
    }


def report_exit_code(report: dict) -> int:
    return 0 if report["passed"] else 1
