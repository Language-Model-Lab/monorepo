"""INT8 round-trip error for every decoder linear of the real checkpoint.
Writes results/int8_weight_error.csv (one row per linear) and results/int8_weight_error.md."""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from quant.loading import MODEL_ID, load_model
from quant.quantize import FP32_SLACK, quantize_int8_per_channel, weight_error_stats

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(torch.bfloat16, device=device)

    rows = []
    for i, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            q, scales = quantize_int8_per_channel(module.weight)
            stats = weight_error_stats(module.weight, q, scales)
            rows.append({
                "layer": i,
                "module": name,
                "N": module.out_features,
                "K": module.in_features,
                "w_absmax": module.weight.abs().max().item(),
                "scale_min": scales.min().item(),
                "scale_median": scales.median().item(),
                "scale_max": scales.max().item(),
                **stats,
            })

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "int8_weight_error.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        lines = [
            f"| {key} | rel. Frobenius error (mean / max) | SQNR dB (min) | max abs err | max err/scale | clipped |",
            "|---|---|---|---|---|---|",
        ]
        for group, members in groups.items():
            rel = [r["rel_fro_err"] for r in members]
            lines.append(
                f"| {group} | {sum(rel) / len(rel):.4%} / {max(rel):.4%} "
                f"| {min(r['sqnr_db'] for r in members):.2f} "
                f"| {max(r['max_abs_err'] for r in members):.3e} "
                f"| {max(r['max_err_over_scale'] for r in members):.4f} "
                f"| {max(r['clipped_frac'] for r in members):.0f} |"
            )
        return "\n".join(lines)

    worst = sorted(rows, key=lambda r: r["rel_fro_err"], reverse=True)[:10]
    max_ratio = max(r["max_err_over_scale"] for r in rows)
    any_clipped = any(r["clipped_frac"] > 0 for r in rows)
    report = "\n\n".join([
        f"# INT8 per-channel weight error: {MODEL_ID}",
        f"{len(rows)} decoder linears, quantized from the BF16 checkpoint weights (symmetric, per output "
        f"channel, round to nearest). Max |error| / scale over all weights: {max_ratio:.7f} "
        f"(bound 0.5 + FP32 rounding slack {FP32_SLACK:.1e}). Any clipping: {any_clipped}.",
        "## By module type",
        summarize("module"),
        "## By layer",
        summarize("layer"),
        "## Ten largest relative errors",
        "| layer | module | rel. Frobenius error | SQNR dB | w absmax | scale max / median |\n|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['layer']} | {r['module']} | {r['rel_fro_err']:.4%} | {r['sqnr_db']:.2f} "
            f"| {r['w_absmax']:.3f} | {r['scale_max'] / r['scale_median']:.1f}x |"
            for r in worst
        ),
    ]) + "\n"
    (RESULTS / "int8_weight_error.md").write_text(report)
    print(report)

    ok = max_ratio <= 0.5 + FP32_SLACK and not any_clipped and all(math.isfinite(r["sqnr_db"]) for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
