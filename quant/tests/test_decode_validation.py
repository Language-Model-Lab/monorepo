"""Gate evaluation, kernel-call coverage and failure paths of the decode validator (K-PLAN-002 stage 3)."""

import copy
import math
from collections import Counter

import pytest
import torch

import quant.gemv
from quant.decode import cached_decode_logits, full_prefix_logits
from quant.decode_validation import (
    COMPARISONS,
    GATES,
    UNKNOWN,
    Gate,
    compare_rows,
    coverage_check,
    evaluate_gates,
    local_check,
    rel_l2,
    report_exit_code,
    run_decode_validation,
    summarize_rows,
)
from quant.quantized_linear import QuantizedLinear
from quant.surgery import quantize_model_, set_backend_

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float16, torch.bfloat16]


#CPU: gate logic

def test_rel_l2_is_defined_for_a_zero_reference():
    zeros = torch.zeros(4)
    assert rel_l2(zeros, zeros) == 0.0
    assert rel_l2(torch.ones(4), zeros) == math.inf
    assert math.isclose(rel_l2(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])), 1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_poison_their_row_and_summary(bad):
    ref = torch.ones(3, 4)
    corrupted = ref.clone()
    corrupted[1, 2] = bad
    for test, reference in ((corrupted, ref), (ref, corrupted)):
        rows = compare_rows(test, reference)
        assert [r["finite"] for r in rows] == [True, False, True] and rows[1]["rel_l2"] == math.inf
        summary = summarize_rows(rows)
        assert not summary["all_finite"] and summary["median"] == summary["max"] == math.inf


def test_summary_statistics():
    rows = [{"finite": True, "rel_l2": v * 1e-3, "max_abs_err": v, "top1_agree": v != 5} for v in (1, 2, 3, 4, 5)]
    s = summarize_rows(rows)
    assert math.isclose(s["median"], 3e-3) and math.isclose(s["p90"], 4.6e-3) and math.isclose(s["max"], 5e-3)
    assert s["top1_agreement"] == 0.8 and s["max_abs_err"] == 5 and s["all_finite"]


GATE = Gate(noise_factor=1.5, noise_bound=1e-1, min_top1=0.98, local_rel_l2=1e-3)


def _summary(median, p90, max_, top1=1.0, finite=True):
    return {"rows": 10, "all_finite": finite, "top1_agreement": top1, "max_abs_err": 0.0,
            "median": median, "p90": p90, "max": max_}


NOISE = _summary(1e-2, 2e-2, 3e-2)


@pytest.mark.parametrize("candidate, passed", [
    (_summary(1.4e-2, 2.8e-2, 4e-2), True),  # within 1.5x of the noise
    (_summary(1.6e-2, 2.8e-2, 4e-2), False),  # median above 1.5x
    (_summary(1.4e-2, 3.2e-2, 4e-2), False),  # p90 above 1.5x
    (_summary(1.4e-2, 2.8e-2, 6e-2), True),  # max 2x the noise's max but under the absolute bound
    (_summary(1.4e-2, 2.8e-2, 2e-1), False),  # max above the absolute bound
    (_summary(1e-2, 2e-2, 3e-2, top1=0.97), False),
    (_summary(math.inf, math.inf, math.inf, finite=False), False),
], ids=["within", "median", "p90", "tail_row", "max_bound", "top1", "non_finite"])
def test_g1_and_g3_are_held_to_the_measured_noise(candidate, passed):
    checks, limits = evaluate_gates({"G1": candidate, "G2": NOISE, "G3": candidate}, GATE)
    assert checks["G1"] is passed and checks["G3"] is passed and checks["G2"]
    assert limits == {"median": 1.5e-2, "p90": 3e-2, "max": GATE.noise_bound}


def test_g2_has_an_absolute_sanity_bound():
    good = _summary(1e-3, 2e-3, 3e-3)
    for noise in (_summary(1e-3, 2e-3, 2e-1), _summary(1e-3, 2e-3, 3e-3, top1=0.9),
                  _summary(math.inf, math.inf, math.inf, finite=False)):
        assert not evaluate_gates({"G1": good, "G2": noise, "G3": good}, GATE)[0]["G2"]


def test_relative_limits_have_the_local_floor_when_noise_is_zero():
    zero = _summary(0.0, 0.0, 0.0)
    assert evaluate_gates({"G1": _summary(5e-4, 9e-4, 1e-3), "G2": zero, "G3": zero}, GATE)[0]["G1"]
    assert not evaluate_gates({"G1": _summary(2e-3, 2e-3, 2e-3), "G2": zero, "G3": zero}, GATE)[0]["G1"]


def test_local_check_needs_every_path_within_its_limit():
    def entry(err, finite=True):
        return {"calls": 3, "max_rel_l2": err, "all_finite": finite}

    assert local_check({"a": entry(5e-4), "b": entry(9e-4)}, ["a", "b"], GATE)["passed"]
    assert not local_check({"a": entry(5e-4)}, ["a", "b"], GATE)["passed"]  # a path never checked
    assert not local_check({"a": entry(5e-4), "b": entry(2e-3)}, ["a", "b"], GATE)["passed"]
    assert not local_check({"a": entry(5e-4), "b": entry(0.0, finite=False)}, ["a", "b"], GATE)["passed"]


def test_coverage_requires_the_exact_path_set_and_counts():
    paths = ["a", "b"]
    assert coverage_check(Counter(a=3, b=3), paths, 3)["passed"]
    fooled = coverage_check(Counter(a=6), paths, 3)  # same total, one path never ran the kernel
    assert fooled["observed_total"] == fooled["expected_total"] and not fooled["passed"]
    assert fooled["missing"] == ["b"] and fooled["wrong_counts"] == {"a": 6}
    assert not coverage_check(Counter(a=3, b=3, c=1), paths, 3)["passed"]
    assert not coverage_check(Counter({"a": 3, "b": 3, UNKNOWN: 1}), paths, 3)["passed"]


def test_fewer_than_two_tokens_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        run_decode_validation(None, None, [torch.arange(3)], 1, GATES[torch.float16])


def test_exit_code_follows_the_report():
    assert report_exit_code({"passed": True}) == 0
    assert report_exit_code({"passed": False}) == 1


#GPU: tiny Qwen3 through both backends

@pytest.fixture(params=DTYPES, ids=str)
def pair(request, tiny_model):
    """(reference model, CUDA model, dtype) sharing identical quantized buffers."""
    dtype = request.param
    model = tiny_model.cuda()
    quantize_model_(model)
    model.to(dtype)
    reference = copy.deepcopy(model)
    set_backend_(reference, "reference")
    set_backend_(model, "cuda")
    return reference, model, dtype


def _prompts(model, n):
    g = torch.Generator().manual_seed(11)
    return [torch.randint(0, model.config.vocab_size, (8,), generator=g) for _ in range(n)]


def _failed(report):
    return {gid for gid in COMPARISONS if not report["checks"][gid]}


def _hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.gpu
@requires_cuda
def test_tiny_qwen_cached_decode_passes_every_gate(pair):
    reference, cuda_model, dtype = pair
    report = run_decode_validation(reference, cuda_model, _prompts(cuda_model, 2), tokens=8, gate=GATES[dtype])
    assert report["passed"], report["checks"]
    assert report["quantized_linear_paths"] == 14
    assert report["matmul_policy"]["cuda_matmul_fp32_precision"] == "ieee"
    for prompt in report["prompts"]:
        coverage = prompt["coverage"]["cuda_cached"]
        assert coverage["observed_paths"] == 14 and coverage["calls_per_path"] == 7
        assert coverage["observed_total"] == 98
        for name in ("ref_cached", "ref_full", "cuda_full"):
            assert prompt["coverage"][name]["observed_total"] == 0
        assert prompt["local"]["passed"] and prompt["local"]["paths"] == 14 and prompt["local"]["calls"] == 98
        assert prompt["prefill_row_bitwise_identical"]
    assert report_exit_code(report) == 0


@pytest.mark.gpu
@requires_cuda
def test_coverage_cannot_be_fooled_by_a_matching_total(pair):
    reference, cuda_model, dtype = pair
    skipped = cuda_model.model.layers[0].mlp.up_proj
    doubled = cuda_model.model.layers[1].self_attn.o_proj
    skipped.backend = "reference"

    def twice(x):
        QuantizedLinear.forward(doubled, x)
        return QuantizedLinear.forward(doubled, x)

    doubled.forward = twice
    report = run_decode_validation(reference, cuda_model, _prompts(cuda_model, 1), tokens=4, gate=GATES[dtype])
    coverage = report["prompts"][0]["coverage"]["cuda_cached"]
    assert coverage["observed_total"] == coverage["expected_total"] == 14 * 3
    assert not coverage["passed"]
    assert coverage["missing"] == ["model.layers.0.mlp.up_proj"]
    assert coverage["wrong_counts"] == {"model.layers.1.self_attn.o_proj": 6}
    assert not report["checks"]["coverage"] and report_exit_code(report) == 1
    assert not report["checks"]["local"]  # the skipped path was never checked against the reference


def _shift_logits(fn):
    return lambda model, prompt, continuation: fn(model, prompt, continuation) + 1.0


def _wrong_tokens(model, prompt, continuation):
    # Always a different token, even for a continuation that repeats one token.
    return cached_decode_logits(model, prompt, (continuation + 1) % model.config.vocab_size)


def _nan_row(model, prompt, continuation):
    logits = cached_decode_logits(model, prompt, continuation)
    logits[1, 0] = float("nan")
    return logits


FAILURES = {
    "cuda_full_corrupted": ({"cuda_full": _shift_logits(full_prefix_logits)}, {"G3"}),
    "ref_full_corrupted": ({"ref_full": _shift_logits(full_prefix_logits)}, {"G2"}),
    "ref_cached_wrong_tokens": ({"ref_cached": _wrong_tokens}, {"G1", "G2"}),
    "nan_in_cuda_cached": ({"cuda_cached": _nan_row}, {"G1", "G3"}),
}


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("name", FAILURES)
def test_each_corruption_fails_its_comparisons(pair, name):
    reference, cuda_model, dtype = pair
    overrides, expected = FAILURES[name]
    report = run_decode_validation(reference, cuda_model, _prompts(cuda_model, 1), tokens=4, gate=GATES[dtype],
                                   overrides=overrides)
    assert _failed(report) == expected
    assert report["checks"]["coverage"] and report["checks"]["local"]  # the kernel itself is fine
    assert report_exit_code(report) == 1


@pytest.mark.gpu
@requires_cuda
def test_perturbed_kernel_output_fails_g1_g3_and_local(pair, monkeypatch):
    reference, cuda_model, dtype = pair
    original = quant.gemv.int8_gemv
    monkeypatch.setattr(quant.gemv, "int8_gemv", lambda x, q, s, b=None: original(x, q, s, b) * 1.1)
    report = run_decode_validation(reference, cuda_model, _prompts(cuda_model, 1), tokens=4, gate=GATES[dtype])
    assert _failed(report) == {"G1", "G3"}
    assert not report["checks"]["local"] and report["checks"]["coverage"] and report_exit_code(report) == 1


@pytest.mark.gpu
@requires_cuda
def test_small_kernel_error_fails_the_local_check(pair, monkeypatch):
    # An error too small to see through the model's noise is still caught per kernel call.
    reference, cuda_model, dtype = pair
    original = quant.gemv.int8_gemv
    bump = 3 * GATES[dtype].local_rel_l2
    monkeypatch.setattr(quant.gemv, "int8_gemv", lambda x, q, s, b=None: original(x, q, s, b) * (1 + bump))
    report = run_decode_validation(reference, cuda_model, _prompts(cuda_model, 1), tokens=4, gate=GATES[dtype])
    assert not report["checks"]["local"] and report_exit_code(report) == 1


@pytest.mark.gpu
@requires_cuda
def test_repeated_runs_leave_no_state_behind(pair):
    reference, cuda_model, dtype = pair
    original = quant.gemv.int8_gemv
    prompts = _prompts(cuda_model, 1)
    first = run_decode_validation(reference, cuda_model, prompts, tokens=3, gate=GATES[dtype])
    second = run_decode_validation(reference, cuda_model, prompts, tokens=3, gate=GATES[dtype])
    assert first["passed"] and second["passed"]
    assert first["prompts"][0]["coverage"] == second["prompts"][0]["coverage"]
    assert quant.gemv.int8_gemv is original
    assert _hook_count(reference) == 0 and _hook_count(cuda_model) == 0
