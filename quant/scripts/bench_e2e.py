""" end-to-end decode timing for one precision per process (plan 6.3-6.4).

For each context length: a non-thinking chat prompt of exactly that many tokens, then `--gen` tokens with
end-of-sequence ignored, seeded sampling with the model-card non-thinking settings. Runs whose SM clock spread
exceeds the threshold are rejected and repeated until `--runs` are accepted or `--max-attempts` is reached.
Writes results/bench/e2e_<precision>[_<tag>].{csv,md,json}."""

import argparse
import sys

import torch

from quant.bench.e2e import SAMPLING_NON_THINKING, build_prompt_ids, decode_run
from quant.bench.report import RESULTS, write_csv, write_json
from quant.bench.stats import summarize
from quant.bench.telemetry import TelemetrySampler, clock_lock_check, environment, is_stable, stability, window
from quant.data import load_wikitext2, tokenize_corpus
from quant.loading import BASELINE_DTYPE, load_int8_model, load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int8_ref"], required=True)
    parser.add_argument("--ctx", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int, default=None, help="per context length; default 2 x runs")
    parser.add_argument("--locked-sm-mhz", type=float, default=None,
                        help="SM clock you locked with 'sudo nvidia-smi -lgc' before this run, to verify it held")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    max_attempts = args.max_attempts or 2 * args.runs

    env = environment()
    if args.precision == "fp16":
        model, tokenizer = load_model(BASELINE_DTYPE)
    else:
        model, tokenizer = load_int8_model(BASELINE_DTYPE)
    torch.cuda.synchronize()
    weights_allocated = torch.cuda.memory_allocated()
    corpus = tokenize_corpus(tokenizer, load_wikitext2("test"))

    rows, summary, run_samples = [], [], []
    with TelemetrySampler() as sampler:
        for ctx in args.ctx:
            prompt = build_prompt_ids(tokenizer, corpus, ctx)
            for _ in range(args.warmup):
                decode_run(model, prompt, args.gen, seed=args.seed)
            accepted, attempt, first_tokens, identical = 0, 0, None, True
            while accepted < args.runs and attempt < max_attempts:
                r = decode_run(model, prompt, args.gen, seed=args.seed)
                sampler.wait_until(r["t1"])
                samples = window(sampler.samples(), r["t0"], r["t1"])
                run_samples += samples
                stab = stability(samples)
                ok = is_stable(stab, args.threshold)
                first_tokens = first_tokens or r["token_ids"]
                identical &= r["token_ids"] == first_tokens
                rows.append({
                    "precision": args.precision, "ctx": ctx, "gen": args.gen, "attempt": attempt, "accepted": ok,
                    **{k: v for k, v in r.items() if k not in ("token_ids", "t0", "t1")},
                    **stab,
                })
                accepted += ok
                attempt += 1
            kept = [row for row in rows if row["ctx"] == ctx and row["accepted"]]
            entry = {"ctx": ctx, "gen": args.gen, "runs_accepted": accepted, "runs_rejected": attempt - accepted,
                     "complete": accepted >= args.runs, "identical_tokens_across_runs": identical}
            if kept:
                ttft = summarize([row["ttft_ms"] for row in kept])
                tok_s = summarize([row["decode_tok_s"] for row in kept])
                entry.update({
                    "ttft_ms_median": ttft["median"], "ttft_ms_iqr": ttft["iqr"],
                    "decode_tok_s_median": tok_s["median"], "decode_tok_s_iqr": tok_s["iqr"],
                    "decode_tok_s_q25": tok_s["q25"], "decode_tok_s_q75": tok_s["q75"],
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in kept),
                    "sm_median_mhz": summarize([row["sm_median_mhz"] for row in kept])["median"],
                })
            summary.append(entry)
            print(f"ctx {ctx}: accepted {accepted}, rejected {attempt - accepted}, "
                  f"decode {entry.get('decode_tok_s_median', float('nan')):.1f} tok/s, "
                  f"TTFT {entry.get('ttft_ms_median', float('nan')):.1f} ms", flush=True)

    clock_lock = clock_lock_check(run_samples, args.locked_sm_mhz)
    stem = f"e2e_{args.precision}" + (f"_{args.tag}" if args.tag else "")
    write_csv(RESULTS / f"{stem}.csv", rows)
    write_json(RESULTS / f"{stem}.json", {
        "environment": env, "args": vars(args), "sampling": SAMPLING_NON_THINKING,
        "weights_allocated_bytes": weights_allocated, "clock_lock": clock_lock, "summary": summary,
    })
    lines = [
        f"# End-to-end decode, {args.precision}" + (f" ({args.tag})" if args.tag else ""),
        f"{env['gpu']}. Prompt: non-thinking chat template filled with WikiText-2 test tokens to exactly the context "
        f"length; {args.gen} generated tokens, end-of-sequence ignored; seeded sampling {SAMPLING_NON_THINKING}. "
        f"Runs with SM clock spread > {args.threshold:.0%} rejected. Power profile: {env['power_profile']}; "
        f"AC power: {env['on_ac_power']}. Allocated after loading: {weights_allocated / 2**30:.3f} GiB. "
        f"Clock lock: {'not attempted' if not clock_lock['attempted'] else ('held' if clock_lock['held'] else 'did not hold')}.",
        "| Context | Accepted / rejected | TTFT median (ms) [IQR] | Decode tok/s median [IQR] | Peak allocated (GiB) | SM clock median (MHz) |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(
            f"| {s['ctx']} | {s['runs_accepted']} / {s['runs_rejected']} | "
            + (f"{s['ttft_ms_median']:.1f} [{s['ttft_ms_iqr']:.1f}] | {s['decode_tok_s_median']:.1f} "
               f"[{s['decode_tok_s_iqr']:.1f}] | {s['peak_allocated_bytes'] / 2**30:.3f} | {s['sm_median_mhz']:.0f} |"
               if s["runs_accepted"] else "– | – | – | – |")
            for s in summary
        ),
    ]
    report = "\n\n".join(lines) + "\n"
    (RESULTS / f"{stem}.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
