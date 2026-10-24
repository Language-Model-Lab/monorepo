"""T choose the 16-bit baseline dtype (BF16 or FP16).

Runs the model in BF16 (as stored) and in an FP16 copy on the same WikiText-2 test chunks. Hooks record
inf/NaN counts and the largest finite |x| at every module output, and at every RMSNorm input (the residual
stream). Rule (plan task 1.5): use BF16 if FP16 produces any inf/NaN or if any activation comes within a
factor of 4 of the FP16 maximum (65,504); otherwise use FP16.
Writes results/dtype_check.json and results/dtype_check.md."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

from quant.data import chat_inputs, load_prompts, load_wikitext2, token_chunks, tokenize_corpus
from quant.loading import MODEL_ID, load_model

RESULTS = Path(__file__).resolve().parents[1] / "results"
FP16_MAX = torch.finfo(torch.float16).max  # 65504
FP16_MIN_NORMAL = torch.finfo(torch.float16).tiny  # 2**-14
FP16_MIN_SUBNORMAL = 2.0**-24
THRESHOLD = FP16_MAX / 4
LOGIT_SLICE = 256  # tokens per slice for the float32 log-softmax, to bound memory
STATS_SLICE = 1 << 24  # elements per slice for activation statistics


def floating_tensors(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from floating_tensors(item)
    elif isinstance(obj, dict):  # ModelOutput is a dict subclass
        for item in obj.values():
            yield from floating_tensors(item)


class ActivationStats:
    """Per hook point: [inf count, NaN count, largest finite |x|], accumulated on the GPU."""

    def __init__(self, model):
        self.stats: dict[str, torch.Tensor] = {}
        self.order: list[str] = []  # forward execution order, to locate where overflow starts
        self.handles = []
        for name, module in model.named_modules():
            if not name:
                continue
            self.handles.append(module.register_forward_hook(self._output_hook(name)))
            if type(module).__name__.endswith("RMSNorm"):
                self.handles.append(module.register_forward_pre_hook(self._input_hook(f"{name} [input]")))

    def _output_hook(self, key):
        return lambda module, args, output: self._update(key, output)

    def _input_hook(self, key):
        return lambda module, args: self._update(key, args)

    @torch.no_grad()
    def _update(self, key, obj):
        for t in floating_tensors(obj):
            # Slices bound the temporaries; on the full logits an int64 isinf().sum() alone needs 1.2 GB.
            for part in t.reshape(-1).split(STATS_SLICE):
                new = torch.stack([
                    torch.isinf(part).sum().double(),
                    torch.isnan(part).sum().double(),
                    part.abs().masked_fill(~torch.isfinite(part), 0).amax().double(),
                ])
                if key not in self.stats:
                    self.stats[key] = new
                    self.order.append(key)
                else:
                    old = self.stats[key]
                    self.stats[key] = torch.stack([old[0] + new[0], old[1] + new[1], torch.maximum(old[2], new[2])])

    def results(self) -> dict[str, dict]:
        out = {}
        for key in self.order:
            inf, nan, absmax = self.stats[key].tolist()
            out[key] = {"inf": int(inf), "nan": int(nan), "absmax": absmax}
        return out

    def remove(self):
        for handle in self.handles:
            handle.remove()


def weight_report(model) -> dict:
    """What casting the BF16 weights to FP16 would do to them."""
    per_tensor = []
    overflow = subnormal = flushed = total = 0
    for name, p in model.named_parameters():
        a = p.detach().float().abs()
        nonzero = a > 0
        overflow += (a > FP16_MAX).sum().item()
        subnormal += (nonzero & (a < FP16_MIN_NORMAL) & (a >= FP16_MIN_SUBNORMAL)).sum().item()
        flushed += (nonzero & (a < FP16_MIN_SUBNORMAL / 2)).sum().item()
        total += a.numel()
        per_tensor.append({"name": name, "absmax": a.max().item()})
    per_tensor.sort(key=lambda r: r["absmax"], reverse=True)
    return {
        "max_abs_weight": per_tensor[0]["absmax"],
        "weights_above_fp16_max": overflow,
        "weights_fp16_subnormal": subnormal,
        "weights_flushed_to_zero_in_fp16": flushed,
        "total_weights": total,
        "largest_tensors": per_tensor[:10],
    }


def merge(stats_bf16: ActivationStats, stats_fp16: ActivationStats) -> list[dict]:
    """One row per hook point, in forward order. Removes both sets of hooks."""
    act_bf16, act_fp16 = stats_bf16.results(), stats_fp16.results()
    stats_bf16.remove()
    stats_fp16.remove()
    return [
        {
            "name": key,
            "bf16_absmax": act_bf16[key]["absmax"],
            "fp16_absmax": act_fp16[key]["absmax"],
            "bf16_inf": act_bf16[key]["inf"],
            "bf16_nan": act_bf16[key]["nan"],
            "fp16_inf": act_fp16[key]["inf"],
            "fp16_nan": act_fp16[key]["nan"],
        }
        for key in stats_bf16.order
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--chunk-len", type=int, default=1024)
    args = parser.parse_args()

    start = time.perf_counter()
    model_bf16, tokenizer = load_model(torch.bfloat16)
    weights = weight_report(model_bf16)
    model_fp16 = copy.deepcopy(model_bf16).to(torch.float16)

    ids = tokenize_corpus(tokenizer, load_wikitext2("test"))
    chunks = token_chunks(ids, args.chunk_len, args.chunks)

    stats_bf16, stats_fp16 = ActivationStats(model_bf16), ActivationStats(model_fp16)
    nll = {"bf16": 0.0, "fp16": 0.0}
    kl_sum = kl_max = max_logit_diff = 0.0
    top1_agree = scored = 0
    chunks_nonfinite_fp16_logits = 0

    with torch.no_grad():
        for chunk in chunks:
            input_ids = chunk[None].cuda()
            logits_b = model_bf16(input_ids, use_cache=False).logits[0]
            logits_h = model_fp16(input_ids, use_cache=False).logits[0]
            chunks_nonfinite_fp16_logits += int(not torch.isfinite(logits_h).all())
            targets = input_ids[0, 1:]
            for s in range(0, targets.numel(), LOGIT_SLICE):
                e = min(s + LOGIT_SLICE, targets.numel())
                lb = logits_b[s:e].float().log_softmax(-1)
                lh = logits_h[s:e].float().log_softmax(-1)
                t = targets[s:e, None]
                nll["bf16"] -= lb.gather(1, t).sum().item()
                nll["fp16"] -= lh.gather(1, t).sum().item()
                kl = (lb.exp() * (lb - lh)).sum(-1)  # KL(BF16 || FP16) per token
                kl_sum += kl.sum().item()
                kl_max = max(kl_max, kl.max().item())
                top1_agree += (lb.argmax(-1) == lh.argmax(-1)).sum().item()
                max_logit_diff = max(
                    max_logit_diff, (logits_b[s:e].float() - logits_h[s:e].float()).abs().max().item()
                )
                scored += e - s

    modules = merge(stats_bf16, stats_fp16)

    # Chat-formatted inputs always start with <|im_start|>; raw-text chunks never do.
    # The largest activation sits at position 0, so these are checked separately.
    prompts = load_prompts()
    chat_stats_bf16, chat_stats_fp16 = ActivationStats(model_bf16), ActivationStats(model_fp16)
    with torch.no_grad():
        for prompt in prompts:
            for thinking in (False, True):
                input_ids = chat_inputs(tokenizer, prompt["text"], thinking)["input_ids"].cuda()
                model_bf16(input_ids, use_cache=False)
                model_fp16(input_ids, use_cache=False)
    chat_modules = merge(chat_stats_bf16, chat_stats_fp16)
    chat_top = sorted(chat_modules, key=lambda m: m["bf16_absmax"], reverse=True)[:5]

    both = modules + chat_modules
    fp16_nonfinite = sum(m["fp16_inf"] + m["fp16_nan"] for m in both)
    first_nonfinite_fp16 = next((m["name"] for m in both if m["fp16_inf"] + m["fp16_nan"]), None)
    bf16_absmax = max(m["bf16_absmax"] for m in both)
    chat_nonfinite = sum(m["fp16_inf"] + m["fp16_nan"] for m in chat_modules)
    reasons = []
    if fp16_nonfinite:
        reasons.append(f"FP16 produced {fp16_nonfinite:,} inf/NaN values (first at {first_nonfinite_fp16})")
    if bf16_absmax > THRESHOLD:
        reasons.append(f"largest BF16 activation {bf16_absmax:,.1f} exceeds 65,504 / 4 = {THRESHOLD:,.0f}")
    if weights["weights_above_fp16_max"]:
        reasons.append(f"{weights['weights_above_fp16_max']} weights exceed the FP16 maximum")
    decision = "bfloat16" if reasons else "float16"

    ppl = {k: math.exp(v / scored) for k, v in nll.items()}
    top = sorted(modules, key=lambda m: m["bf16_absmax"], reverse=True)[:20]
    residual = [m for m in modules if m["name"].count(".") == 2 and m["name"].startswith("model.layers.")]
    result = {
        "model_id": MODEL_ID,
        "decision": decision,
        "reasons": reasons,
        "rule": "BF16 if FP16 has any inf/NaN or any |activation| > 65504/4; otherwise FP16",
        "data": {"dataset": "wikitext-2-raw-v1 test", "chunks": args.chunks, "chunk_len": args.chunk_len,
                 "tokens_scored": scored},
        "perplexity": ppl,
        "logits": {
            "kl_bf16_fp16_mean": kl_sum / scored,
            "kl_bf16_fp16_max": kl_max,
            "top1_agreement": top1_agree / scored,
            "max_abs_logit_diff": max_logit_diff,
            "chunks_with_nonfinite_fp16_logits": chunks_nonfinite_fp16_logits,
        },
        "activations": {
            "bf16_absmax": bf16_absmax,
            "fp16_nonfinite_total": fp16_nonfinite,
            "first_nonfinite_fp16_module": first_nonfinite_fp16,
            "top20_by_bf16_absmax": top,
        },
        "chat": {
            "inputs": f"{len(prompts)} prompts from prompts.json x enable_thinking in (False, True)",
            "bf16_absmax": chat_top[0]["bf16_absmax"],
            "fp16_nonfinite_total": chat_nonfinite,
            "top5_by_bf16_absmax": chat_top,
        },
        "weights": weights,
        "seconds": time.perf_counter() - start,
        "modules": modules,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dtype_check.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# Baseline dtype check: {MODEL_ID}",
        f"Decision: **{decision}**. " + ("; ".join(reasons) + "." if reasons else "No rule triggered."),
        f"Data: {args.chunks} non-overlapping chunks of {args.chunk_len} tokens from WikiText-2 test, "
        f"{scored:,} scored tokens. Rule: {result['rule']}.",
        "## Logits and perplexity",
        "| Metric | Value |\n|---|---|\n"
        f"| Perplexity, BF16 | {ppl['bf16']:.4f} |\n"
        f"| Perplexity, FP16 | {ppl['fp16']:.4f} |\n"
        f"| KL(BF16 ‖ FP16), mean / max per token | {kl_sum / scored:.3e} / {kl_max:.3e} |\n"
        f"| Top-1 agreement | {top1_agree / scored:.4%} |\n"
        f"| Max \\|logit difference\\| | {max_logit_diff:.4f} |\n"
        f"| Chunks with non-finite FP16 logits | {chunks_nonfinite_fp16_logits} |",
        "## Activations",
        f"Largest finite |activation| in BF16, WikiText-2 and chat inputs: {bf16_absmax:,.1f} "
        f"(threshold {THRESHOLD:,.0f}). Non-finite values in FP16: {fp16_nonfinite:,}. "
        f"First FP16 module with non-finite output: {first_nonfinite_fp16}.",
        "### Chat-template inputs",
        f"{len(prompts)} prompts from prompts.json, each with thinking off and on. Position 0 is always "
        f"<|im_start|>. Largest BF16 |activation|: {chat_top[0]['bf16_absmax']:,.1f} at {chat_top[0]['name']}. "
        f"Non-finite values in FP16: {chat_nonfinite:,}.",
        "### WikiText-2: top 20 hook points by BF16 |activation|",
        "| Hook point | BF16 absmax | FP16 absmax | FP16 inf | FP16 NaN |\n|---|---|---|---|---|\n"
        + "\n".join(
            f"| {m['name']} | {m['bf16_absmax']:,.1f} | {m['fp16_absmax']:,.1f} | {m['fp16_inf']:,} | {m['fp16_nan']:,} |"
            for m in top
        ),
        "### WikiText-2: residual stream (decoder layer outputs)",
        "| Layer | BF16 absmax | FP16 absmax | FP16 inf + NaN |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {m['name'].split('.')[2]} | {m['bf16_absmax']:,.1f} | {m['fp16_absmax']:,.1f} "
            f"| {m['fp16_inf'] + m['fp16_nan']:,} |"
            for m in residual
        ),
        "## Weights cast to FP16",
        f"Largest |weight|: {weights['max_abs_weight']:.4f}. Above FP16 max: {weights['weights_above_fp16_max']}. "
        f"In FP16 subnormal range: {weights['weights_fp16_subnormal']:,} of {weights['total_weights']:,}. "
        f"Flushed to zero: {weights['weights_flushed_to_zero_in_fp16']:,}.",
        "| Tensor | Largest \\|w\\| |\n|---|---|\n"
        + "\n".join(f"| {t['name']} | {t['absmax']:.4f} |" for t in weights["largest_tensors"][:5]),
    ]
    report = "\n\n".join(lines) + "\n"
    (RESULTS / "dtype_check.md").write_text(report)
    print(report)
    print(f"({result['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
