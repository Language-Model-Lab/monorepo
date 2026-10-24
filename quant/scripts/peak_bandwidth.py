""" measured peak DRAM bandwidth, the denominator for kernel efficiency (plan section 6.3).

Methods: the custom streaming-read kernel (csrc/bandwidth.cu), a device-to-device copy (bytes read + written),
and torch.sum. Buffer sizes from below to far above the 24 MiB L2, each measured two ways:
  warm: the same buffer every call, so buffers that fit in L2 are read from L2;
  cold: calls rotate through enough buffer copies (at least 2 x L2 between reuses) that every call reads DRAM.
Calls are timed in batches (time_cuda_rotating): CUDA event readings cluster at multiples of 1.024 us, too
coarse for single short calls. Writes results/bench/peak_bandwidth.{csv,md,json} and peak_bandwidth_summary.csv."""

import argparse
import sys

import torch

from quant.bench.bandwidth import stream_read
from quant.bench.report import RESULTS, write_csv, write_json
from quant.bench.runner import timed_block
from quant.bench.stats import gbs, summarize
from quant.bench.telemetry import TelemetrySampler, environment
from quant.bench.timing import rotation_count, time_cuda_rotating

SPEC_GBS = 192.0  # RTX 4050 Laptop spec sheet: 16 Gbps GDDR6 x 96-bit bus
METHODS = ("read_kernel", "copy", "torch_sum")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 256, 1024])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    env = environment()
    l2 = env["l2_bytes"]
    rows, summary = [], []
    with TelemetrySampler() as sampler:
        for size_mib in args.sizes_mib:
            nbytes = size_mib << 20
            for method in METHODS:
                moved = 2 * nbytes if method == "copy" else nbytes
                # Copies: enough distinct buffers for cold reads; also the batch size for both modes.
                copies = 1 if moved >= 2 * l2 else rotation_count(moved, l2)
                # Random contents: zero-filled buffers can read faster on GPUs with data compression.
                srcs = [torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda") for _ in range(copies)]
                dsts = [torch.empty_like(src) for src in srcs] if method == "copy" else []
                if method == "read_kernel":
                    calls = [lambda src=src: stream_read(src) for src in srcs]
                elif method == "copy":
                    calls = [lambda src=src, dst=dst: dst.copy_(src) for src, dst in zip(srcs, dsts)]
                else:
                    calls = [lambda src=src: torch.sum(src.view(torch.float32)) for src in srcs]
                for cache in ("warm", "cold"):
                    fns = [calls[0]] * copies if cache == "warm" else calls
                    attempts = timed_block(
                        lambda: time_cuda_rotating(fns, warmup=args.warmup, iters=args.iters),
                        sampler, threshold=args.threshold, max_attempts=args.max_attempts,
                    )
                    config = {"method": method, "size_mib": size_mib, "cache": cache, "bytes_moved": moved,
                              "calls_per_sample": copies}
                    for a in attempts:
                        for i, ms in enumerate(a["times_ms"]):
                            rows.append({**config, "attempt": a["attempt"], "accepted": a["accepted"], "iter": i,
                                         "time_ms": ms, "gbs": gbs(moved, ms)})
                    final = attempts[-1]
                    t = summarize(final["times_ms"])
                    bw = summarize([gbs(moved, ms) for ms in final["times_ms"]])
                    summary.append({
                        **config,
                        "median_ms": t["median"],
                        "iqr_ms": t["iqr"],
                        "median_gbs": bw["median"],
                        "gbs_q25": bw["q25"],
                        "gbs_q75": bw["q75"],
                        "accepted": final["accepted"],
                        "attempts": len(attempts),
                        "block_s": final["block_s"],
                        **final["stability"],
                    })
                    print(f"{method:12s} {size_mib:5d} MiB {cache:4s} x{copies:<3d} {bw['median']:7.1f} GB/s "
                          f"(attempts {len(attempts)}, accepted {final['accepted']})", flush=True)
                del srcs, dsts, calls

    def pick(method, size_mib, cache="cold"):
        return next(s for s in summary if s["method"] == method and s["size_mib"] == size_mib and s["cache"] == cache)

    largest = max(args.sizes_mib)
    headline = {
        "size_mib": largest,
        "peak_read_gbs": pick("read_kernel", largest)["median_gbs"],
        "peak_copy_gbs": pick("copy", largest)["median_gbs"],
        "torch_sum_gbs": pick("torch_sum", largest)["median_gbs"],
        "spec_gbs": SPEC_GBS,
        "nvml_theoretical_gbs": env["theoretical_bw_gbs"],
    }
    for key in ("peak_read_gbs", "peak_copy_gbs", "torch_sum_gbs"):
        headline[f"{key}_pct_of_spec"] = headline[key] / SPEC_GBS
        headline[f"{key}_pct_of_nvml"] = headline[key] / env["theoretical_bw_gbs"]

    write_csv(RESULTS / "peak_bandwidth.csv", rows)
    write_csv(RESULTS / "peak_bandwidth_summary.csv", summary)
    write_json(RESULTS / "peak_bandwidth.json", {"environment": env, "args": vars(args), "headline": headline,
                                                  "summary": summary})

    lines = [
        "# Measured peak bandwidth",
        f"{env['gpu']}, driver {env['driver']}, torch {env['torch']}. Power profile: {env['power_profile']}; "
        f"AC power: {env['on_ac_power']}. Warnings: {', '.join(env['warnings']) or 'none'}.",
        f"Theoretical: {SPEC_GBS:.0f} GB/s (spec sheet), {env['theoretical_bw_gbs']:.1f} GB/s (NVML: maximum memory "
        f"clock {env['max_mem_mhz']} MHz x 2 x {env['memory_bus_bits']}-bit bus).",
        "## Headline (largest buffer)",
        "| Method | GB/s | % of spec | % of NVML theoretical |\n|---|---|---|---|\n" + "\n".join(
            f"| {name} | {headline[key]:.1f} | {headline[key + '_pct_of_spec']:.1%} | {headline[key + '_pct_of_nvml']:.1%} |"
            for name, key in (("Streaming read (peak)", "peak_read_gbs"), ("Device-to-device copy", "peak_copy_gbs"),
                              ("torch.sum", "torch_sum_gbs"))
        ),
        "## By buffer size (median GB/s; IQR in brackets)",
        "Warm: same buffer every call (L2-resident when it fits). Cold: rotating copies, every call reads DRAM. "
        "Copy counts bytes read + written.",
        "| Size (MiB) | Read, warm | Read, cold | Copy, warm | Copy, cold | torch.sum, warm | torch.sum, cold |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {size} | " + " | ".join(
                f"{s['median_gbs']:.1f} [{s['gbs_q25']:.0f}–{s['gbs_q75']:.0f}]"
                for s in (pick(m, size, c) for m in METHODS for c in ("warm", "cold"))
            ) + " |"
            for size in args.sizes_mib
        ),
        "## Run stability",
        f"{sum(s['accepted'] for s in summary)} of {len(summary)} configurations accepted "
        f"(SM clock spread <= {args.threshold:.0%}); {sum(s['attempts'] - 1 for s in summary)} blocks retried. "
        f"SM clock median range: {min(s['sm_median_mhz'] for s in summary)}–{max(s['sm_median_mhz'] for s in summary)} MHz. "
        f"Memory clock median range: {min(s['mem_median_mhz'] for s in summary)}–{max(s['mem_median_mhz'] for s in summary)} MHz. "
        f"Power-capped blocks: {sum(s['power_capped'] for s in summary)}. Thermal-slowdown blocks: "
        f"{sum(s['thermal_slowdown'] for s in summary)}.",
    ]
    report = "\n\n".join(lines) + "\n"
    (RESULTS / "peak_bandwidth.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
