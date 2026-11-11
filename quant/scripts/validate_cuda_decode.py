"""Real-model validation of CUDA cached decode (K-PLAN-002 stage 4).

Loads the INT8 model once with the reference backend, deep-copies it to a CUDA-backend twin with identical
buffers, and runs gates G1-G3 and per-path kernel coverage on the eight packaged prompts (non-thinking chat
template), all under the IEEE FP32 matmul policy. A seeded side-by-side generation is recorded as a
diagnostic only.

Exits non-zero if any gate fails, if the source aggregate differs from --expect-aggregate, or if the source
changes during the run. Writes results/cuda_decode_validation_<dtype>.{json,md}."""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from torch.utils.cpp_extension import CUDA_HOME

from quant.bench.report import write_json
from quant.bench.telemetry import environment
from quant.data import chat_inputs, load_prompts
from quant.decode_validation import COMPARISONS, GATES, report_exit_code, run_decode_validation
from quant.fidelity import generate_side_by_side
from quant.loading import MODEL_ID, load_int8_model
from quant.precision import ieee_fp32_matmul
from quant.provenance import source_identity
from quant.quantized_linear import QuantizedLinear
from quant.surgery import set_backend_

RESULTS = Path(__file__).resolve().parents[1] / "results"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
EXPECTED_LINEARS = 28 * 7


#The commit of the cached Hugging Face snapshot that `filename` was loaded from (.../snapshots/<commit>/...).
def _snapshot_commit(filename: str, revision: str | None) -> str | None:
    path = try_to_load_from_cache(MODEL_ID, filename, revision=revision)
    return Path(path).parent.name if isinstance(path, str) else None


def _nvcc_version() -> str | None:
    try:
        out = subprocess.run([str(Path(CUDA_HOME) / "bin" / "nvcc"), "--version"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError, TypeError):
        return None
    return out.strip().splitlines()[-2] if out.strip() else None


def _verdict(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def _markdown(result: dict) -> str:
    report, dtype = result["report"], result["args"]["dtype"]
    gate, limits, pooled = report["gate"], report["relative_limits"], report["pooled"]
    local = [p["local"] for p in report["prompts"]]
    worst_local = max(local, key=lambda entry: entry["max_rel_l2"])
    lines = [
        f"# CUDA cached-decode validation, {dtype}",
        f"**{'PASSED' if result['passed'] else 'FAILED'}.** Checks: "
        + ", ".join(f"{k} {_verdict(v)}" for k, v in result["checks"].items()) + ".",
        f"Model {MODEL_ID} at commit {result['provenance']['model_commit']}; INT8 per-channel weights, "
        f"{report['quantized_linear_paths']} QuantizedLinear modules; {len(report['prompts'])} prompts, "
        f"{report['tokens']} continuation tokens (greedy from the reference). Matmul policy: "
        f"{report['matmul_policy']}.",
        f"Gates (noise-relative, decision of 2026-10-28): G2, the reference model's own cache-path noise, "
        f"must have max relative L2 <= {gate['noise_bound']} and top-1 agreement >= {gate['min_top1']}. G1 and "
        f"G3 median and p90 must be at most max({gate['noise_factor']} x G2's, {gate['local_rel_l2']}), their max "
        f"<= {gate['noise_bound']}, and top-1 agreement >= {gate['min_top1']}. Every kernel call must be within "
        f"relative L2 {gate['local_rel_l2']} of the reference expression on the same input.",
        f"Source aggregate {result['provenance']['source_start']['aggregate_sha256']} "
        f"(expected {result['args']['expect_aggregate']}); unchanged during run: "
        f"{not result['provenance']['source_changed_during_run']}.",
        "## Pooled over all prompts and rows",
        "| Gate | Comparison | Median | p90 | Max | Top-1 | Limits (median / p90 / max) | Verdict |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {g} | {a} vs {b} | {pooled[g]['median']:.2e} | {pooled[g]['p90']:.2e} | {pooled[g]['max']:.2e} "
            f"| {pooled[g]['top1_agreement']:.3f} | "
            + ("– / – / " + f"{gate['noise_bound']:.0e}" if g == "G2" else
               f"{limits['median']:.2e} / {limits['p90']:.2e} / {limits['max']:.0e}")
            + f" | {_verdict(report['checks'][g])} |"
            for g, (a, b) in COMPARISONS.items()
        ),
        "## Kernel checks",
        f"Coverage: {sum(p['coverage']['cuda_cached']['observed_total'] for p in report['prompts'])} kernel calls in "
        f"the CUDA cached passes, {report['quantized_linear_paths']} paths x {report['tokens'] - 1} calls per prompt "
        f"expected; zero calls in every reference and full-prefix pass: {_verdict(report['checks']['coverage'])}. "
        f"Local: {sum(entry['calls'] for entry in local)} calls checked, max relative L2 "
        f"{worst_local['max_rel_l2']:.2e} at {worst_local['worst_path']} (limit {gate['local_rel_l2']}): "
        f"{_verdict(report['checks']['local'])}.",
        "## By prompt (median / max relative L2)",
        "| Prompt | P | G1 | G2 | G3 | Kernel calls | Local max | Prefill bitwise | Smoke first divergence |\n"
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for prompt_id, entry, smoke in zip(result["prompt_ids"], report["prompts"], result["smoke"]):
        s, cov, divergence = entry["summaries"], entry["coverage"]["cuda_cached"], smoke["first_divergence"]["int8_cuda"]
        lines.append(
            f"| {prompt_id} | {entry['prompt_tokens']} | "
            + " | ".join(f"{s[g]['median']:.1e} / {s[g]['max']:.1e}" for g in COMPARISONS)
            + f" | {cov['observed_total']} / {cov['expected_total']} | {entry['local']['max_rel_l2']:.1e} "
            f"| {entry['prefill_row_bitwise_identical']} | {'identical' if divergence is None else divergence} |"
        )
    return "\n\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--revision", default=None, help="checkpoint commit to load; record it for evidence")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--smoke-tokens", type=int, default=32)
    parser.add_argument("--expect-aggregate", default=None, help="frozen source aggregate the run must match")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 so that decode steps run")
    dtype = DTYPES[args.dtype]

    source_start = source_identity()
    if args.expect_aggregate is not None and source_start["aggregate_sha256"] != args.expect_aggregate:
        print(f"source aggregate {source_start['aggregate_sha256']} != expected {args.expect_aggregate}",
              file=sys.stderr)
        return 2

    env = environment()
    model_ref, tokenizer = load_int8_model(dtype, backend="reference", revision=args.revision)
    model_cuda = copy.deepcopy(model_ref)
    switched = set_backend_(model_cuda, "cuda")
    ref_buffers = dict(model_ref.named_buffers())
    identical = switched == EXPECTED_LINEARS and all(
        torch.equal(buffer, ref_buffers[name])
        for name, buffer in model_cuda.named_buffers()
        if name.endswith((".qweight", ".scales"))
    )
    if not identical:
        print(f"CUDA twin is not identical to the reference ({switched} QuantizedLinear modules)", file=sys.stderr)
        return 1

    prompts = load_prompts()
    inputs = [chat_inputs(tokenizer, p["text"], enable_thinking=False) for p in prompts]
    report = run_decode_validation(model_ref, model_cuda, [i["input_ids"][0] for i in inputs], args.tokens,
                                   GATES[dtype])
    with ieee_fp32_matmul():
        smoke = generate_side_by_side({"int8_ref": model_ref, "int8_cuda": model_cuda}, inputs, seed=0,
                                      max_new_tokens=args.smoke_tokens)

    source_end = source_identity()
    changed = source_end["aggregate_sha256"] != source_start["aggregate_sha256"]
    checks = {
        **report["checks"],
        "source_matches_expected": args.expect_aggregate is None
        or source_start["aggregate_sha256"] == args.expect_aggregate,
        "source_unchanged_during_run": not changed,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "args": {**vars(args), "out_dir": str(args.out_dir)},
        "environment": env,
        "provenance": {
            "model_id": MODEL_ID,
            "requested_revision": args.revision,
            "model_commit": getattr(model_ref.config, "_commit_hash", None) or _snapshot_commit("config.json", args.revision),
            "tokenizer_commit": _snapshot_commit("tokenizer.json", args.revision),
            "source_start": source_start,
            "source_end": source_end,
            "source_changed_during_run": changed,
            "nvcc": _nvcc_version(),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "quantized_linear_modules": sum(isinstance(m, QuantizedLinear) for m in model_cuda.modules()),
        },
        "prompt_ids": [p["id"] for p in prompts],
        "report": report,
        "smoke": [{"first_divergence": row["first_divergence"], "token_ids": row["token_ids"]} for row in smoke],
    }
    stem = args.out_dir / f"cuda_decode_validation_{args.dtype}"
    write_json(stem.with_suffix(".json"), result)
    markdown = _markdown(result)
    stem.with_suffix(".md").write_text(markdown)
    print(markdown)
    return report_exit_code(result)  # gates, coverage and source checks together


if __name__ == "__main__":
    sys.exit(main())
