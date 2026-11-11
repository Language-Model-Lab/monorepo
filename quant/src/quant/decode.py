"""Teacher-forced decode helpers: logits for a fixed continuation, through the KV cache or in one pass.

For a prompt of P tokens and a continuation of L tokens, row i of every result is the logit vector at query
position P - 1 + i, which predicts the token at position P + i, i.e. continuation[i]. Row 0 comes from the
prompt's own forward pass; in cached_decode_logits, rows 1 .. L-1 come from single-token decode steps, step i
feeding continuation[i - 1] at position P + i - 1.

Inputs are 1-D int64 token tensors (batch 1) with P >= 1 and L >= 1. Results are FP32 [L, vocab].
"""

from collections.abc import Callable

import torch


def _check_ids(name: str, ids: torch.Tensor) -> None:
    if not isinstance(ids, torch.Tensor) or ids.dim() != 1:
        raise ValueError(f"{name} must be a 1-D token tensor, got {getattr(ids, 'shape', type(ids))}")
    if ids.dtype != torch.int64:
        raise ValueError(f"{name} must be int64, got {ids.dtype}")
    if ids.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def _decode(model, prompt_ids: torch.Tensor, steps: int, next_token: Callable[[int, torch.Tensor], torch.Tensor]):
    """Prefill, then `steps` single-token steps through the cache. next_token(i, last_row) is the token fed
    at step i + 1. Returns the FP32 rows, prefill row first."""
    device = model.device
    out = model(input_ids=prompt_ids.to(device)[None], use_cache=True, logits_to_keep=1)
    rows = [out.logits[0, -1].float()]
    for i in range(steps):
        token = next_token(i, rows[-1]).to(device).view(1, 1)
        out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
        rows.append(out.logits[0, -1].float())
    return rows


@torch.no_grad()
def cached_decode_logits(model, prompt_ids: torch.Tensor, continuation: torch.Tensor) -> torch.Tensor:
    """Rows for continuation through the KV cache; L = 1 runs the prefill only."""
    _check_ids("prompt_ids", prompt_ids)
    _check_ids("continuation", continuation)
    return torch.stack(_decode(model, prompt_ids, continuation.numel() - 1, lambda i, _: continuation[i]))


@torch.no_grad()
def full_prefix_logits(model, prompt_ids: torch.Tensor, continuation: torch.Tensor) -> torch.Tensor:
    """The same rows from one uncached pass over prompt + continuation[:-1] (query positions P-1 .. P+L-2)."""
    _check_ids("prompt_ids", prompt_ids)
    _check_ids("continuation", continuation)
    p, length = prompt_ids.numel(), continuation.numel()
    ids = torch.cat([prompt_ids, continuation[:-1]]).to(model.device)[None]
    return model(input_ids=ids, use_cache=False).logits[0, p - 1 : p + length - 1].float()


@torch.no_grad()
def greedy_continuation(model, prompt_ids: torch.Tensor, n: int) -> torch.Tensor:
    """n argmax tokens through the cache, ignoring end-of-sequence. Returns int64 [n] on the CPU."""
    _check_ids("prompt_ids", prompt_ids)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tokens = []

    def pick(i, row):
        tokens.append(row.argmax())
        return tokens[-1]

    rows = _decode(model, prompt_ids, n - 1, pick)
    tokens.append(rows[-1].argmax())
    return torch.stack(tokens).cpu()