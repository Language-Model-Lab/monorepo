"""JIT builds from CUDA sources bundled with the quant package."""

import os
from functools import cache
from importlib.resources import as_file, files

from torch.utils.cpp_extension import load

@cache
def load_extension(name: str, sources: tuple[str, ...]):
    # Build for the target architecture only: shorter builds, unambiguous SASS for Nsight Compute.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    # Keep the entire source directory available until compilation finishes,
    # including when resources need extraction from an archive.
    with as_file(files("quant").joinpath("csrc")) as csrc:
        return load(
            name=name,
            sources=[str(csrc / s) for s in sources],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )


def bandwidth_ext():
    return load_extension("quant_bandwidth", ("bandwidth.cu",))
