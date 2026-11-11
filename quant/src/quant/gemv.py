"""Python entry points for the v0 batch-1 GEMV kernels in csrc/gemv.cu.

The extension is built on first call. Metadata (dtype, shape, contiguity, device) is validated at the C++
boundary; this layer only enforces the inference-only contract, since the kernels have no backward.
"""

import torch


def _ext():
    # Imported here so that importing quant does not import torch.utils.cpp_extension.
    from quant import ext

    return ext.gemv_ext()


def _check_inference(*tensors) -> None:
    # Grad mode by itself is fine; an input that requires grad would silently lose its gradient.
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise RuntimeError(
            "the CUDA GEMV kernels are inference-only: run under torch.no_grad() or pass inputs that do not "
            "require grad"
        )


#y [1, N] = cast(scales * (x @ qweight^T) + bias), FP32 accumulation; x [1, K] fp16/bf16, qweight int8.
def int8_gemv(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    _check_inference(x, qweight, scales, bias)
    return _ext().int8_gemv(x, qweight, scales, bias)


#y [1, N] = cast(x @ weight^T + bias), FP32 accumulation; the 16-bit control with int8_gemv's structure.
def gemv_16bit_control(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    _check_inference(x, weight, bias)
    return _ext().gemv_16bit_control(x, weight, bias)
