from collections.abc import Callable

import torch
from torch import nn

from quant.quantized_linear import QuantizedLinear

#Raises unless lm_head still shares its tensor with embed_tokens.
def verify_tied_embeddings(model) -> None:
    out_w = model.get_output_embeddings().weight
    in_w = model.get_input_embeddings().weight
    if out_w.data_ptr() != in_w.data_ptr():
        raise RuntimeError("lm_head no longer shares its weight with embed_tokens")


@torch.no_grad()
def quantize_model_(
    model,
    backend: str = "reference",
    predicate: Callable[[str], bool] | None = None,
) -> list[str]:
    """Replaces every nn.Linear inside the decoder layers with QuantizedLinear, in place.

    Module paths and attribute names are unchanged, so hooks registered by path work the same on
    16-bit and quantized models. lm_head is outside the decoder layers and is never touched.
    `predicate(path)` optionally restricts which linears are replaced. Returns the replaced paths.
    """
    targets = []
    for i, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            path = f"model.layers.{i}.{name}"
            if isinstance(module, nn.Linear) and (predicate is None or predicate(path)):
                targets.append((layer, name, path, module))

    # Check everything before replacing anything, so a refusal leaves the model untouched.
    for _, _, path, module in targets:
        if module.weight.dtype == torch.float16:
            raise ValueError(
                f"{path} is FP16; load the model in BF16, run surgery, then cast the model"
            )

    for layer, name, _, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = layer.get_submodule(parent_name) if parent_name else layer
        setattr(parent, attr, QuantizedLinear.from_linear(module, backend=backend))

    if getattr(model.config, "tie_word_embeddings", False):
        verify_tied_embeddings(model)
    return [path for _, _, path, _ in targets]
