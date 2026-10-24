"""record the model's config, linear layers and tied weights, and check them against
the tables in plan section 4.1. Writes results/inventory.json; exits non-zero on any mismatch."""

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from torch import nn

from quant.loading import MODEL_ID, load_model

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Plan section 4.1: (N, K) = (out_features, in_features)
EXPECTED_SHAPES = {
    "self_attn.q_proj": (2048, 1024),
    "self_attn.k_proj": (1024, 1024),
    "self_attn.v_proj": (1024, 1024),
    "self_attn.o_proj": (1024, 2048),
    "mlp.gate_proj": (3072, 1024),
    "mlp.up_proj": (3072, 1024),
    "mlp.down_proj": (1024, 3072),
}
EXPECTED_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "head_dim": 128,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "attention_bias": False,
}
EXPECTED_DECODER_LINEAR_PARAMS = 28 * sum(n * k for n, k in EXPECTED_SHAPES.values())  # 440,401,920
EXPECTED_EMBEDDING_PARAMS = 151936 * 1024  # 155,582,464

CONFIG_FIELDS = [*EXPECTED_CONFIG, "rope_parameters", "rms_norm_eps", "max_position_embeddings", "dtype"]


def main() -> int:
    model, _ = load_model(torch.bfloat16, device="cpu")
    config = model.config
    failures = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    config_values = {field: getattr(config, field, None) for field in CONFIG_FIELDS}
    config_values["dtype"] = str(config_values["dtype"])
    for field, expected in EXPECTED_CONFIG.items():
        check(config_values[field] == expected, f"config.{field} = {config_values[field]}, expected {expected}")

    linears = [
        {
            "path": name,
            "N": m.out_features,
            "K": m.in_features,
            "bias": m.bias is not None,
            "dtype": str(m.weight.dtype),
        }
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    decoder_linears = [lin for lin in linears if lin["path"].startswith("model.layers.")]
    check(len(decoder_linears) == 28 * 7, f"{len(decoder_linears)} decoder linears, expected 196")
    for lin in decoder_linears:
        suffix = lin["path"].split(".", 3)[3]  # drop "model.layers.<i>."
        expected = EXPECTED_SHAPES.get(suffix)
        check(expected == (lin["N"], lin["K"]), f"{lin['path']} is {(lin['N'], lin['K'])}, expected {expected}")
        check(not lin["bias"], f"{lin['path']} has a bias")
        check(lin["K"] % 512 == 0 and lin["K"] % 128 == 0, f"{lin['path']} K={lin['K']} not a multiple of 512")

    shape_counts = Counter((lin["path"].split(".", 3)[3], lin["N"], lin["K"]) for lin in decoder_linears)

    lm_head, embed = model.get_output_embeddings(), model.get_input_embeddings()
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()
    check(tied, "lm_head and embed_tokens do not share a tensor")

    qk_norms = []
    for i, layer in enumerate(model.model.layers):
        for name in ("q_norm", "k_norm"):
            norm = getattr(layer.self_attn, name, None)
            shape = tuple(norm.weight.shape) if norm is not None else None
            check(shape == (config.head_dim,), f"layer {i} {name} has shape {shape}")
            qk_norms.append({"layer": i, "name": name, "shape": shape})

    total_params = sum(p.numel() for p in model.parameters())  # tied tensor counted once
    decoder_linear_params = sum(lin["N"] * lin["K"] for lin in decoder_linears)
    embedding_params = embed.weight.numel()
    check(
        decoder_linear_params == EXPECTED_DECODER_LINEAR_PARAMS,
        f"decoder linear params {decoder_linear_params:,}, expected {EXPECTED_DECODER_LINEAR_PARAMS:,}",
    )
    check(
        embedding_params == EXPECTED_EMBEDDING_PARAMS,
        f"embedding params {embedding_params:,}, expected {EXPECTED_EMBEDDING_PARAMS:,}",
    )

    inventory = {
        "model_id": MODEL_ID,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "config": config_values,
        "params": {
            "total": total_params,
            "decoder_linears": decoder_linear_params,
            "embedding_lm_head_shared": embedding_params,
            "other": total_params - decoder_linear_params - embedding_params,
        },
        "lm_head_tied_to_embed_tokens": tied,
        "unique_decoder_linear_shapes": [
            {"module": module, "N": n, "K": k, "count": count} for (module, n, k), count in shape_counts.items()
        ],
        "non_decoder_linears": [lin for lin in linears if lin not in decoder_linears],
        "qk_norms_per_layer": sorted({(q["name"], q["shape"]) for q in qk_norms}),
        "decoder_linears": decoder_linears,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    print(f"# Inventory: {MODEL_ID}\n")
    print(f"torch {torch.__version__}, transformers {transformers.__version__}\n")
    print("| Config field | Value |\n|---|---|")
    for field, value in config_values.items():
        print(f"| {field} | {value} |")
    print("\n| Module | N | K | Count |\n|---|---|---|---|")
    for (module, n, k), count in shape_counts.items():
        print(f"| {module} | {n} | {k} | {count} |")
    print(f"\nParameters: total {total_params:,}; decoder linears {decoder_linear_params:,}; "
          f"embedding / LM head {embedding_params:,}")
    print(f"lm_head tied to embed_tokens: {tied}")
    print(f"q_norm / k_norm on every layer: {sum(q['shape'] == (config.head_dim,) for q in qk_norms)}/56")

    if failures:
        print(f"\n{len(failures)} check(s) FAILED:")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
