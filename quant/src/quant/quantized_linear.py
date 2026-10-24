import torch
import torch.nn.functional as F
from torch import nn

from quant.quantize import quantize_int8_per_channel

BACKENDS = ("reference", "cuda")


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear holding an INT8 per-channel weight (W8A16).

    The reference backend computes y = s * (x @ q^T) (+ b) in FP32 and casts to the input dtype:
    the exact math of plan section 5.1, and the FP32 reference the CUDA kernel is tested against.
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
            y = F.linear(x.float(), self.qweight.float()) * self.scales
            if self.bias is not None:
                y = y + self.bias.float()
            return y.to(x.dtype)
        if self.backend == "cuda":
            raise NotImplementedError("the CUDA backend arrives in Phase 2 (task 2.K6)")
        raise ValueError(f"unknown backend {self.backend!r}")

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
