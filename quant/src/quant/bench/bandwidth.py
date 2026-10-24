import numpy as np
import torch

from quant.ext import bandwidth_ext

#Reads every byte of x once on the GPU; returns per-block XOR partials. Does not synchronize.
def stream_read(x: torch.Tensor, blocks: int | None = None, threads: int = 256) -> torch.Tensor:
    if blocks is None:
        blocks = torch.cuda.get_device_properties(x.device).multi_processor_count * 32
    return bandwidth_ext().stream_read(x, blocks, threads)

#XOR of the partials: the XOR of all little-endian 32-bit words of the input, zero padded.
def xor_checksum(partials: torch.Tensor) -> int:
    values = partials.cpu().numpy().view(np.uint32)
    return int(np.bitwise_xor.reduce(values)) if values.size else 0
