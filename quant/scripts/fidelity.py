""" fidelity ladder rungs 2-4, FP16 baseline vs INT8-reference (BF16 as a yardstick in rungs 3-4).

Rung 2: per-layer activations on fixed WikiText-2 sequences, compounded and local, position 0 separate.
Rung 3: WikiText-2 test perplexity (sliding window) and logit agreement with the FP16 baseline.
Rung 4: seeded generations on prompts.json, non-thinking mode.
Writes results/fidelity_int8_ref.json and results/fidelity_int8_ref.md."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from quant.data import chat_inputs, load_prompts, load_wikitext2, token_chunks, tokenize_corpus
from quant.fidelity import compare_activations, generate_side_by_side, sliding_window_eval
from quant.loading import BASELINE_DTYPE, MODEL_ID, load_int8_model, load_model

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "residual",
]


def split_point(path: str) -> tuple[int, str]:
    parts = path.split(".", 3)
    return int(parts[2]), parts[3] if len(parts) > 3 else "residual"


def by_module(stats: dict, region: str = "rest") -> dict[str, list[tuple[int, dict]]]:
    groups: dict[str, list[tuple[int, dict]]] = {}
    for path, regions in stats.items():
        layer, module = split_point(path)
        groups.setdefault(module, []).append((layer, regions[region]))
    return groups


def module_table(rung2: dict) -> str:
    comp, loc = by_module(rung2["compounded"]), by_module(rung2["local"])
    lines = [
        "| Module | Compounded rel. L2, mean / max (layer) | Compounded cos, mean / min | "
        "Local rel. L2, mean / max (layer) | Local cos, min |",
        "|---|---|---|---|---|",
    ]
    for module in MODULE_ORDER:
        if module not in comp:
            continue
        c = comp[module]
        c_max = max(c, key=lambda x: x[1]["rel_l2"])
        cells = [
            f"{statistics.mean(s['rel_l2'] for _, s in c):.3%} / {c_max[1]['rel_l2']:.3%} ({c_max[0]})",
            f"{statistics.mean(s['cos_mean'] for _, s in c):.6f} / {min(s['cos_min'] for _, s in c):.6f}",
        ]
        if module in loc:
            lc = loc[module]
            l_max = max(lc, key=lambda x: x[1]["rel_l2"])
            cells += [
                f"{statistics.mean(s['rel_l2'] for _, s in lc):.3%} / {l_max[1]['rel_l2']:.3%} ({l_max[0]})",
                f"{min(s['cos_min'] for _, s in lc):.6f}",
            ]
        else:
            cells += ["n/a", "n/a"]
        lines.append(f"| {module} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def residual_table(rung2: dict) -> str:
    rest = dict(by_module(rung2["compounded"], "rest")["residual"])
    pos0 = dict(by_module(rung2["compounded"], "pos0")["residual"])
    lines = [
        "| Layer | Rel. L2, pos >= 1 | Cos mean / min, pos >= 1 | Rel. L2, pos 0 | Max \\|x\\| pos 0, FP16 / INT8 |",
        "|---|---|---|---|---|",
    ]
    for layer in sorted(rest):
        r, p = rest[layer], pos0[layer]
        lines.append(
            f"| {layer} | {r['rel_l2']:.3%} | {r['cos_mean']:.6f} / {r['cos_min']:.6f} | {p['rel_l2']:.3%} "
            f"| {p['ref_absmax']:,.1f} / {p['test_absmax']:,.1f} |"
        )
    return "\n".join(lines)


def absorption(rung2: dict) -> tuple[str, dict]:
    summary, lines = {}, [
        "| Projection | Mode | Rel. L2 before norm, mean | After norm, mean | Ratio after / before, mean (min – max over layers) |",
        "|---|---|---|---|---|",
    ]
    for proj, norm in (("self_attn.q_proj", "self_attn.q_norm"), ("self_attn.k_proj", "self_attn.k_norm")):
        for mode in ("local", "compounded"):
            groups = by_module(rung2[mode])
            before, after = dict(groups[proj]), dict(groups[norm])
            ratios = [after[i]["rel_l2"] / before[i]["rel_l2"] for i in sorted(before)]
            entry = {
                "before_mean": statistics.mean(s["rel_l2"] for s in before.values()),
                "after_mean": statistics.mean(s["rel_l2"] for s in after.values()),
                "ratio_mean": statistics.mean(ratios),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "ratio_per_layer": ratios,
            }
            summary[f"{proj.split('.')[-1]}_{mode}"] = entry
            lines.append(
                f"| {proj.split('.')[-1]} | {mode} | {entry['before_mean']:.3%} | {entry['after_mean']:.3%} "
                f"| {entry['ratio_mean']:.3f} ({entry['ratio_min']:.3f} – {entry['ratio_max']:.3f}) |"
            )
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    timings = {}

    model_fp, tokenizer = load_model(BASELINE_DTYPE)
    model_q, _ = load_int8_model(BASELINE_DTYPE)
    ids = tokenize_corpus(tokenizer, load_wikitext2("test"))

    start = time.perf_counter()
    batches = [chunk[None] for chunk in token_chunks(ids, args.seq_len, args.seqs)]
    rung2 = compare_activations(model_fp, model_q, batches)
    timings["rung2_s"] = time.perf_counter() - start
    print(f"rung 2 done ({timings['rung2_s']:.0f} s)", flush=True)

    models = {"fp16": model_fp, "int8_ref": model_q}
    if not args.no_bf16:
        models["bf16"], _ = load_model(torch.bfloat16)

    start = time.perf_counter()
    rung3 = sliding_window_eval(models, ids, ctx=args.ctx, stride=args.stride, max_tokens=args.max_tokens)
    timings["rung3_s"] = time.perf_counter() - start
    print(f"rung 3 done ({timings['rung3_s']:.0f} s)", flush=True)

    start = time.perf_counter()
    prompts = load_prompts()
    inputs_list = [chat_inputs(tokenizer, p["text"], enable_thinking=False) for p in prompts]
    rows = generate_side_by_side(models, inputs_list, seed=args.seed, max_new_tokens=args.max_new_tokens)
    for prompt, row in zip(prompts, rows):
        row["id"], row["prompt"] = prompt["id"], prompt["text"]
        row["text"] = {
            name: tokenizer.decode(token_ids, skip_special_tokens=True) for name, token_ids in row["token_ids"].items()
        }
    timings["rung4_s"] = time.perf_counter() - start

    absorption_md, absorption_summary = absorption(rung2)
    down2 = rung2["compounded"]["model.layers.2.mlp.down_proj"]["pos0"]
    down2_local = rung2["local"]["model.layers.2.mlp.down_proj"]["pos0"]
    result = {
        "model_id": MODEL_ID,
        "baseline_dtype": str(BASELINE_DTYPE),
        "args": vars(args),
        "timings": timings,
        "rung2": {"sequences": args.seqs, "seq_len": args.seq_len, **rung2},
        "qk_norm_absorption": absorption_summary,
        "rung3": rung3,
        "rung4": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fidelity_int8_ref.json").write_text(json.dumps(result, indent=2) + "\n")

    ppl = rung3["models"]
    ref_ppl = ppl["fp16"]["perplexity"]
    ppl_lines = [
        "| Model | Perplexity | Change vs FP16 | KL(FP16 ‖ model), mean / max | Top-1 agreement | Max \\|logit diff\\| |",
        "|---|---|---|---|---|---|",
        f"| fp16 (baseline) | {ref_ppl:.4f} | – | – | – | – |",
    ]
    for name, vs in rung3["vs_reference"].items():
        p = ppl[name]["perplexity"]
        ppl_lines.append(
            f"| {name} | {p:.4f} | {(p - ref_ppl) / ref_ppl:+.3%} | {vs['kl_mean']:.3e} / {vs['kl_max']:.3e} "
            f"| {vs['top1_agreement']:.3%} | {vs['max_abs_logit_diff']:.3f} |"
        )

    gen_lines = ["| Prompt | First token where output differs from FP16 |", "|---|---|"]
    for row in rows:
        div = ", ".join(
            f"{name}: {'identical' if idx is None else idx}" for name, idx in row["first_divergence"].items()
        )
        gen_lines.append(f"| {row['id']} | {div} |")
    gen_text = []
    for row in rows:
        gen_text.append(f"#### {row['id']}: {row['prompt']}")
        for name, text in row["text"].items():
            quoted = "\n".join("> " + line for line in text.strip().splitlines()) or "> (empty)"
            gen_text.append(f"**{name}**\n\n{quoted}")

    report = "\n\n".join([
        f"# Fidelity, FP16 baseline vs INT8-reference: {MODEL_ID}",
        f"Baseline and activation dtype: {BASELINE_DTYPE}. INT8: symmetric per-channel, round to nearest, "
        "reference backend (FP32 math). BF16 is included in rungs 3–4 as a yardstick for a change with no "
        "quantization.",
        "## Rung 3: WikiText-2 test perplexity",
        f"Sliding window, context {rung3['ctx']}, stride {rung3['stride']}, {rung3['windows']} windows, "
        f"{rung3['scored']:,} scored tokens, raw text without the chat template.",
        "\n".join(ppl_lines),
        "## Rung 2: per-layer activations",
        f"{args.seqs} WikiText-2 test sequences of {args.seq_len} tokens. Compounded: INT8 model against FP16 "
        "model, each on its own forward pass. Local: each INT8 linear on the FP16 model's input to that linear. "
        "Positions >= 1 only; position 0 is reported separately below.",
        "### By module type (positions >= 1)",
        module_table(rung2),
        "### Q/K norm absorption (positions >= 1)",
        absorption_md,
        "### Residual stream by layer (compounded)",
        residual_table(rung2),
        "### Position 0 at layer 2 down_proj",
        f"Largest |x|: FP16 {down2['ref_absmax']:,.1f}, INT8 {down2['test_absmax']:,.1f}. "
        f"Relative L2 error: compounded {down2['rel_l2']:.3%}, local {down2_local['rel_l2']:.3%}.",
        f"## Rung 4: generations ({len(rows)} prompts, non-thinking, seeded sampling, "
        f"{args.max_new_tokens} new tokens)",
        "\n".join(gen_lines),
        *gen_text,
    ]) + "\n"
    (RESULTS / "fidelity_int8_ref.md").write_text(report)
    print(report)
    print(f"timings: {timings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
