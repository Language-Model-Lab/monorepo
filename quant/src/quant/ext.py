"""JIT builds of the CUDA sources in quant/csrc ."""

import os
from functools import cache
from pathlib import Path

from torch.utils.cpp_extension import load

CSRC = Path(__file__).resolve().parents[2] / "csrc"


@cache
def load_extension(name: str, sources: tuple[str, ...]):
    # Build for the target architecture only: shorter builds, unambiguous SASS for Nsight Compute.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    return load(
        name=name,
        sources=[str(CSRC / s) for s in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def bandwidth_ext():
    return load_extension("quant_bandwidth", ("bandwidth.cu",))
