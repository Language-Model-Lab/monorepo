"""Explicit FP32 matmul policy for numerical oracles.

torch 2.14 has a legacy precision API (torch.set_float32_matmul_precision, matmul.allow_tf32) and a new one
(fp32_precision). They must not be mixed: once the new API has been used, the legacy getters raise. This
module therefore reads and writes only the new API.
"""

from contextlib import contextmanager

import torch


#The settings that decide how CUDA matmuls round, read through the new API only.
def matmul_policy() -> dict:
    matmul = torch.backends.cuda.matmul
    return {
        "cuda_matmul_fp32_precision": matmul.fp32_precision,
        "generic_fp32_precision": torch.backends.fp32_precision,
        "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }


@contextmanager
def ieee_fp32_matmul():
    """IEEE FP32 (no TF32) for CUDA FP32 matmuls inside the block. Yields the effective policy and restores
    the previous setting on exit, including after an exception. 16-bit reduction flags are left as they are.
    """
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    matmul.fp32_precision = "ieee"
    try:
        yield matmul_policy()
    finally:
        matmul.fp32_precision = previous