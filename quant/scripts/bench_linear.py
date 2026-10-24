""" batch-1 linear layers at the five decode shapes (plan section 4.1).

Calls rotate through enough copies of the weights (at least 2 x L2 between reuses) that every call reads them
from DRAM, and are timed in batches of one rotation per CUDA event pair (event readings cluster at multiples of
1.024 us, too coarse for single 10-40 us calls). Effective bandwidth uses the plan
section 6.3 bytes model (weights + scales + x + y) and is reported against the measured peak read bandwidth
from results/bench/peak_bandwidth.json. Backends:
  cublas_fp16: F.linear with FP16 weights (cuBLAS), the external reference.
  int8_ref:    QuantizedLinear, reference backend (dequantize to FP32, then FP32 matmul).
Writes results/bench/bench_linear_<backend>.{csv,md,json}."""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from torch import nn

from quant.bench.bandwidth import stream_read
from quant.bench.report import RESULTS, write_csv, write_json
from quant.bench.runner import timed_block
from quant.bench.stats import bytes_gemv, gbs, summarize
from quant.bench.telemetry import TelemetrySampler, environment
from quant.bench.timing import rotation_count, time_cuda_rotating
from quant.quantized_linear import QuantizedLinear

SHAPES = {  # (N, K) = (out_features, in_features)
    "q_proj": (2048, 1024),
    "k_proj, v_proj": (1024, 1024),
    "o_proj": (1024, 2048),
    "gate_proj, up_proj": (3072, 1024),
    "down_proj": (1024, 3072),
}


def bytes_moved(backend: str, n: int, k: int) -> int:
    if backend == "cublas_fp16":
        return bytes_gemv(n, k, 2, 2)
    if backend == "int8_ref":
        return bytes_gemv(n, k, 1, 2, scale_bytes=4 * n)
    raise ValueError(f"unknown backend {backend!r}")


#One batch 1 linear call with its own weights and input.
def make_call(backend: str, n: int, k: int, seed: int):
    torch.manual_seed(seed)
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    weight = torch.randn(n, k, device="cuda") * 0.02
    if backend == "cublas_fp16":
        w16 = weight.half()
        return lambda: F.linear(x, w16)
    linear = nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    module = QuantizedLinear.from_linear(linear).half()
    return lambda: module(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cublas_fp16", "int8_ref"], default="cublas_fp16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    peak_path = RESULTS / "peak_bandwidth.json"
    if not peak_path.exists():
        print(f"{peak_path} not found; run scripts/peak_bandwidth.py first", file=sys.stderr)
        return 1
    peak = json.loads(peak_path.read_text())["headline"]["peak_read_gbs"]

    env = environment()
    rows, summary = [], []
    with TelemetrySampler() as sampler:
        for name, (n, k) in SHAPES.items():
            moved = bytes_moved(args.backend, n, k)
            copies = rotation_count(moved, env["l2_bytes"])
            calls = [make_call(args.backend, n, k, seed) for seed in range(copies)]
            attempts = timed_block(lambda: time_cuda_rotating(calls, warmup=args.warmup, iters=args.iters), sampler,
                                   threshold=args.threshold, max_attempts=args.max_attempts)
            config = {"backend": args.backend, "shape": name, "N": n, "K": k, "bytes_moved": moved,
                      "calls_per_sample": copies}
            for a in attempts:
                for i, ms in enumerate(a["times_ms"]):
                    rows.append({**config, "attempt": a["attempt"], "accepted": a["accepted"], "iter": i,
                                 "time_us": ms * 1000, "gbs": gbs(moved, ms)})
            final = attempts[-1]
            t = summarize([ms * 1000 for ms in final["times_ms"]])
            median_gbs = gbs(moved, t["median"] / 1000)
            del calls
            # Size-matched reference: the read kernel on rotating buffers of the same byte count, timed the same way.
            # Short transfers do not reach the large-buffer peak, so this is the achievable rate at this size.
            buffers = [torch.randint(0, 256, (moved,), dtype=torch.uint8, device="cuda") for _ in range(copies)]
            reads = [lambda b=b: stream_read(b) for b in buffers]
            ref = timed_block(lambda: time_cuda_rotating(reads, warmup=args.warmup, iters=args.iters), sampler,
                              threshold=args.threshold, max_attempts=args.max_attempts)[-1]
            matched_gbs = gbs(moved, summarize(ref["times_ms"])["median"])
            del buffers, reads
            summary.append({
                **config,
                "median_us": t["median"],
                "iqr_us": t["iqr"],
                "median_gbs": median_gbs,
                "pct_of_peak": median_gbs / peak,
                "matched_read_gbs": matched_gbs,
                "pct_of_matched": median_gbs / matched_gbs,
                "accepted": final["accepted"],
                "attempts": len(attempts),
                "block_s": final["block_s"],
                **final["stability"],
            })
            print(f"{name:20s} {t['median']:8.1f} us  {median_gbs:6.1f} GB/s  {median_gbs / peak:6.1%} of peak  "
                  f"{median_gbs / matched_gbs:6.1%} of size-matched read ({matched_gbs:.1f} GB/s)", flush=True)

    stem = f"bench_linear_{args.backend}"
    write_csv(RESULTS / f"{stem}.csv", rows)
    write_json(RESULTS / f"{stem}.json", {"environment": env, "args": vars(args), "peak_read_gbs": peak,
                                          "summary": summary})
    lines = [
        f"# Batch-1 linear layers, backend {args.backend}",
        f"{env['gpu']}. Weights read from DRAM on every call (rotating copies), timed in batches of one rotation. Peak read bandwidth (measured): {peak:.1f} GB/s. "
        f"Power profile: {env['power_profile']}; AC power: {env['on_ac_power']}.",
        "Size-matched read: the streaming-read kernel on rotating buffers of the same byte count, timed the same way; the "
        "achievable bandwidth at that transfer size.",
        "| Shape | N | K | Bytes moved | Median (µs) | IQR (µs) | Effective GB/s | % of measured peak "
        "| Size-matched read GB/s | % of size-matched | Accepted |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {s['shape']} | {s['N']} | {s['K']} | {s['bytes_moved']:,} | {s['median_us']:.1f} | {s['iqr_us']:.1f} "
            f"| {s['median_gbs']:.1f} | {s['pct_of_peak']:.1%} | {s['matched_read_gbs']:.1f} | {s['pct_of_matched']:.1%} "
            f"| {s['accepted']} |"
            for s in summary
        ),
    ]
    report = "\n\n".join(lines) + "\n"
    (RESULTS / f"{stem}.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
