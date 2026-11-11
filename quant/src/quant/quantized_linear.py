import math

import torch
import torch.nn.functional as F
from torch import nn

from quant import gemv as _gemv
from quant.quantize import quantize_int8_per_channel

BACKENDS = ("reference", "cuda")
CUDA_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear holding an INT8 per-channel weight (W8A16).

    The reference backend computes y = s * (x @ q^T) (+ b) in FP32 and casts to the input dtype:
    the exact math of plan section 5.1, and the FP32 reference the CUDA kernel is tested against.

    The cuda backend runs single-vector inputs (batch-1 decode) through the v0 INT8 GEMV kernel and every
    other input through the reference arithmetic, so prefill keeps the reference numerics. It is
    inference-only and needs CUDA tensors with FP16 or BF16 activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        backend: str = "reference",
        device=None,
        bias_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scales", torch.empty(out_features, dtype=torch.float32, device=device))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=bias_dtype, device=device))
        else:
            self.register_buffer("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, backend: str = "reference") -> "QuantizedLinear":
        qweight, scales = quantize_int8_per_channel(linear.weight)
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            backend=backend,
            device=linear.weight.device,
            bias_dtype=linear.weight.dtype,
        )
        module.qweight.copy_(qweight)
        module.scales.copy_(scales)
        if linear.bias is not None:
            module.bias.copy_(linear.bias)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "reference":
            return self._reference_forward(x)
        if self.backend == "cuda":
            return self._cuda_forward(x)
        raise ValueError(f"unknown backend {self.backend!r}")

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x.float(), self.qweight.float()) * self.scales
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype)

    def _cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got shape {tuple(x.shape)}")
        # Never fall back to CPU execution silently: a CUDA backend on CPU tensors is a configuration error.
        if not (x.is_cuda and self.qweight.is_cuda and self.scales.is_cuda):
            raise RuntimeError(
                f"backend='cuda' needs CUDA tensors; got x on {x.device}, qweight on {self.qweight.device}, "
                f"scales on {self.scales.device}"
            )
        if x.dtype not in CUDA_ACTIVATION_DTYPES:
            raise RuntimeError(f"backend='cuda' needs float16 or bfloat16 activations, got {x.dtype}")
        # One contract for both paths below, although only the kernel lacks a backward.
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                "backend='cuda' is inference-only: run under torch.no_grad() or pass inputs that do not "
                "require grad"
            )
        leading = x.shape[:-1]
        vectors = math.prod(leading)
        if vectors == 0:
            return x.new_empty((*leading, self.out_features))
        if vectors > 1:
            return self._reference_forward(x)
        bias = self.bias.float() if self.bias is not None else None
        y = _gemv.int8_gemv(x.reshape(1, self.in_features).contiguous(), self.qweight, self.scales, bias)
        return y.view(*leading, self.out_features)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype) and .half() would also cast the FP32 scales. Hide them from fn,
        # then move them to wherever fn sends FP32 data, keeping their dtype.
        scales = self._buffers["scales"]
        self._buffers["scales"] = None
        try:
            return super()._apply(fn, recurse)
        finally:
            device = fn(torch.empty(0, dtype=torch.float32, device=scales.device)).device
            self._buffers["scales"] = scales.to(device)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend}"
        )
