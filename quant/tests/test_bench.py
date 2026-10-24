import math
import statistics
import time

import numpy as np
import pytest
import torch

from quant.bench.e2e import build_prompt_ids, sample_next
from quant.bench.report import read_csv, write_csv, write_json
from quant.bench.stats import bytes_gemv, gbs, summarize
from quant.bench.telemetry import (
    REASON_SW_POWER_CAP,
    REASON_SW_THERMAL,
    Sample,
    clock_lock_check,
    is_stable,
    stability,
    window,
)

def _samples(sm_values, start=0.0, step=0.1, reasons=0):
    return [Sample(start + i * step, sm, 7825, 50, 30.0, reasons) for i, sm in enumerate(sm_values)]


#CPU:

def test_summarize_known_values():
    s = summarize([4, 1, 3, 2, 5])
    assert s["n"] == 5 and s["median"] == 3 and s["min"] == 1 and s["max"] == 5
    assert s["q25"] == 2 and s["q75"] == 4 and s["iqr"] == 2
    assert math.isclose(s["mean"], 3) and math.isclose(s["std"], statistics.stdev([1, 2, 3, 4, 5]))


def test_summarize_matches_numpy_quartiles():
    values = np.random.default_rng(0).normal(size=51).tolist()
    s = summarize(values)
    assert math.isclose(s["q25"], np.percentile(values, 25)) and math.isclose(s["q75"], np.percentile(values, 75))


def test_summarize_single_value():
    assert summarize([2.5])["iqr"] == 0


def test_bytes_model_decode_shapes():
    # FP16 q_proj: 2048 x 1024 weights + x (1024) + y (2048), all 2 bytes
    assert bytes_gemv(2048, 1024, 2, 2) == 2048 * 1024 * 2 + (1024 + 2048) * 2
    # INT8 down_proj: 1 byte weights, FP32 scale per row, FP16 activations
    assert bytes_gemv(1024, 3072, 1, 2, scale_bytes=1024 * 4) == 1024 * 3072 + 1024 * 4 + (3072 + 1024) * 2
    assert math.isclose(gbs(2_000_000, 1.0), 2.0)


def test_stability_rejects_large_clock_spread():
    assert not is_stable(stability(_samples([2000, 2200, 2100])), threshold=0.05)  # 10% spread
    assert is_stable(stability(_samples([2000, 2010, 2020])), threshold=0.05)  # 1% spread
    assert not is_stable(stability([]))


def test_stability_flags_power_and_thermal_reasons():
    stab = stability(_samples([2000, 2000], reasons=REASON_SW_POWER_CAP))
    assert stab["power_capped"] and not stab["thermal_slowdown"]
    assert stability(_samples([2000], reasons=REASON_SW_THERMAL))["thermal_slowdown"]


def test_window_inside_and_padded():
    samples = _samples([1, 2, 3, 4, 5, 6])  # t = 0.0 .. 0.5
    assert [s.sm_mhz for s in window(samples, 0.15, 0.45)] == [3, 4, 5]
    # A run shorter than the interval: no samples inside, so the neighbors on each side
    assert [s.sm_mhz for s in window(samples, 0.21, 0.24)] == [3, 4]


def test_clock_lock_check():
    assert clock_lock_check(_samples([2100, 2100, 2101]), 2100)["held"]
    assert not clock_lock_check(_samples([2300, 2310]), 2100)["held"]
    assert clock_lock_check([], None) == {"requested_mhz": None, "attempted": False}


def test_sample_next_top_k_one_is_argmax():
    logits = torch.randn(3, 50)
    token = sample_next(logits, torch.Generator().manual_seed(0), temperature=0.7, top_k=1, top_p=0.8)
    assert torch.equal(token[:, 0], logits.argmax(-1))


def test_sample_next_is_seeded():
    logits = torch.randn(4, 100)
    a = sample_next(logits, torch.Generator().manual_seed(7), temperature=0.7, top_k=20, top_p=0.8)
    b = sample_next(logits, torch.Generator().manual_seed(7), temperature=0.7, top_k=20, top_p=0.8)
    assert torch.equal(a, b)


def test_sample_next_respects_top_k_and_top_p():
    # Token probabilities after softmax: 0.5, 0.3, 0.15, 0.05 (plus negligible tail)
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05] + [1e-9] * 20)
    logits = probs.log()[None].repeat(2000, 1)
    gen = torch.Generator().manual_seed(0)
    drawn = set(sample_next(logits, gen, temperature=1.0, top_k=20, top_p=0.8)[:, 0].tolist())
    assert drawn == {0, 1}  # 0.5 + 0.3 reaches top_p = 0.8
    drawn = set(sample_next(logits, gen, temperature=1.0, top_k=3, top_p=1.0)[:, 0].tolist())
    assert drawn == {0, 1, 2}


def test_rotation_count_covers_twice_l2():
    from quant.bench.timing import rotation_count

    l2 = 24 << 20
    assert rotation_count(4 << 20, l2) == 13  # 12 other copies = 48 MiB between reuses of one copy
    assert rotation_count(6_299_648, l2) == 9
    assert rotation_count(64 << 20, l2) == 2


class _FakeSampler:
    """Samples every 0.1 s since construction, at a fixed clock."""

    def __init__(self, sm_mhz=2400):
        self.start, self.sm_mhz = time.monotonic(), sm_mhz

    def samples(self):
        n = int((time.monotonic() - self.start) / 0.1) + 1
        return _samples([self.sm_mhz] * n, start=self.start)

    def wait_until(self, t):
        while self.samples()[-1].t <= t:
            time.sleep(0.01)


def test_timed_block_extends_short_measurements_to_span_samples():
    from quant.bench.runner import timed_block

    calls = []
    attempts = timed_block(lambda: calls.append(1) or [0.01] * 5, _FakeSampler(), min_duration_s=0.35)
    final = attempts[-1]
    assert final["accepted"] and final["block_s"] >= 0.35
    assert len(final["times_ms"]) > 5 and len(final["times_ms"]) % 5 == 0
    assert final["stability"]["n_samples"] >= 3
    assert len(calls) == len(final["times_ms"]) // 5 + 1  # one untimed priming call


def test_csv_and_json_round_trip(tmp_path):
    rows = [{"a": 1, "b": 2.5}, {"a": 3, "c": "x"}]
    write_csv(tmp_path / "r.csv", rows)
    back = read_csv(tmp_path / "r.csv")
    assert list(back[0]) == ["a", "b", "c"]
    assert back[0] == {"a": "1", "b": "2.5", "c": ""} and back[1] == {"a": "3", "b": "", "c": "x"}
    write_json(tmp_path / "r.json", {"x": [1, 2]})
    assert (tmp_path / "r.json").read_text().startswith("{")


#GPU:

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_time_cuda_scales_with_sleep_cycles():
    from quant.bench.timing import time_cuda

    short = statistics.median(time_cuda(lambda: torch.cuda._sleep(1_000_000), warmup=3, iters=10))
    long = statistics.median(time_cuda(lambda: torch.cuda._sleep(2_000_000), warmup=3, iters=10))
    assert short > 0 and 1.7 < long / short < 2.3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("nbytes", [16, 4096, 4096 + 7, (1 << 20) + 13, 64 << 20])
def test_stream_read_checksum(nbytes):
    from quant.bench.bandwidth import stream_read, xor_checksum

    data = np.random.default_rng(nbytes).integers(0, 256, size=nbytes, dtype=np.uint8)
    x = torch.from_numpy(data).cuda()
    padded = np.concatenate([data, np.zeros(-nbytes % 4, dtype=np.uint8)])
    expected = int(np.bitwise_xor.reduce(padded.view("<u4")))
    assert xor_checksum(stream_read(x)) == expected


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_l2_flush_slows_repeated_small_reads():
    from quant.bench.bandwidth import stream_read
    from quant.bench.timing import L2Flusher, time_cuda

    x = torch.zeros(4 << 20, dtype=torch.uint8, device="cuda")  # 4 MiB, fits in L2
    warm = statistics.median(time_cuda(lambda: stream_read(x), iters=30))
    cold = statistics.median(time_cuda(lambda: stream_read(x), iters=30, flush=L2Flusher()))
    assert cold > 1.3 * warm


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rotating_timer_reads_cold_and_resolves_below_timer_step():
    from quant.bench.bandwidth import stream_read
    from quant.bench.timing import rotation_count, time_cuda_rotating

    nbytes = 4 << 20
    copies = rotation_count(nbytes)
    buffers = [torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda") for _ in range(copies)]
    cold = time_cuda_rotating([lambda b=b: stream_read(b) for b in buffers], iters=20)
    warm = time_cuda_rotating([lambda: stream_read(buffers[0])] * copies, iters=20)
    assert statistics.median(cold) > 1.3 * statistics.median(warm)
    # Single-call event readings cluster at multiples of 1.024 us; batched per-call times do not.
    steps = [t * 1000 / 1.024 for t in cold]
    assert any(abs(s - round(s)) > 0.05 for s in steps)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_telemetry_sampler_collects_plausible_samples():
    from quant.bench.telemetry import TelemetrySampler

    with TelemetrySampler(interval_ms=100) as sampler:
        t0 = time.monotonic()
        time.sleep(0.5)
        t1 = time.monotonic()
    samples = window(sampler.samples(), t0, t1)
    assert len(samples) >= 3
    assert all(0 < s.sm_mhz < 5000 and 0 < s.mem_mhz < 20000 and 0 < s.temp_c < 110 for s in samples)
    assert all(0 <= s.power_w < 500 for s in samples)


#real tokenizer:

@pytest.mark.model
@pytest.mark.parametrize("length", [512, 1024, 4096])
def test_build_prompt_ids_exact_length(length):
    from transformers import AutoTokenizer

    from quant.data import load_wikitext2, tokenize_corpus
    from quant.loading import MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    corpus = tokenize_corpus(tokenizer, load_wikitext2("test"))
    ids = build_prompt_ids(tokenizer, corpus, length)
    assert ids.shape == (length,)
    text = tokenizer.decode(ids)
    assert text.startswith("<|im_start|>user\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
