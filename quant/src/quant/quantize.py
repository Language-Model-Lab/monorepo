"""Symmetric per-output-channel INT8 weight quantization 

For W of shape [N, K]:
    s_n    = max_k |W[n,k]| / 127
    q[n,k] = clamp(round(W[n,k] / s_n), -127, 127)
torch.round rounds half to even. -128 is never produced, so the range is symmetric.
"""

import math

import torch

QMAX = 127
# Measured |error| / scale can exceed 0.5 by FP32 rounding in q*s - w, with |q*s| up to 127 s.
FP32_SLACK = QMAX * 2.0**-23


#Returns (qweight int8 [N, K] contiguous, scales fp32 [N]).
def quantize_int8_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    # An FP16 copy of a BF16 checkpoint may already be rounded or clipped; quantize from the source.
    if w.dtype == torch.float16:
        raise ValueError("refusing to quantize an FP16 weight; quantize from the BF16 checkpoint weights")
    if w.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported weight dtype {w.dtype}")

    w32 = w.detach().to(torch.float32)
    amax = w32.abs().amax(dim=1)
    # An all-zero row would give a zero scale; any nonzero scale quantizes it to zeros.
    scales = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.round(w32 / scales[:, None]).clamp_(-QMAX, QMAX).to(torch.int8)
    return q.contiguous(), scales.contiguous()


def dequantize_int8_per_channel(
    q: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return (q.to(torch.float32) * scales[:, None]).to(dtype)


def weight_error_stats(w: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    w32 = w.detach().to(torch.float32)
    err = dequantize_int8_per_channel(q, scales) - w32
    abs_err = err.abs()
    signal = w32.pow(2).sum().item()
    noise = err.pow(2).sum().item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_fro_err": math.sqrt(noise / signal) if signal > 0 else 0.0,
        "sqnr_db": 10 * math.log10(signal / noise) if noise > 0 else math.inf,
        # Round-to-nearest without clipping keeps every element within half a step.
        "max_err_over_scale": (abs_err / scales[:, None]).max().item(),
        "clipped_frac": (w32.abs() / scales[:, None] > QMAX + 0.5).float().mean().item(),
    }
