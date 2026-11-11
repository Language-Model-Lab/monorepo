import pytest
import torch

from quant.precision import ieee_fp32_matmul, matmul_policy


@pytest.fixture
def tf32_policy():
    """Start from a policy that differs from the oracle's, then restore whatever was set before."""
    matmul = torch.backends.cuda.matmul
    original = matmul.fp32_precision
    matmul.fp32_precision = "tf32"
    yield
    matmul.fp32_precision = original


def test_selects_ieee_and_restores(tf32_policy):
    with ieee_fp32_matmul() as policy:
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert policy["cuda_matmul_fp32_precision"] == "ieee"
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"


def test_restores_after_exception(tf32_policy):
    with pytest.raises(RuntimeError, match="inside"):
        with ieee_fp32_matmul():
            raise RuntimeError("inside")
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"


def test_policy_reads_only_the_new_api(tf32_policy):
    # The legacy getters raise once the new API has been used; matmul_policy must not call them.
    policy = matmul_policy()
    assert policy["cuda_matmul_fp32_precision"] == "tf32"
    assert set(policy) == {
        "cuda_matmul_fp32_precision",
        "generic_fp32_precision",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    }
