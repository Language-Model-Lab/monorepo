"""QuantizedLinear(backend="cuda") dispatch (K-PLAN-002 stage 3): one vector runs the kernel, anything else
the unchanged reference arithmetic."""

import pytest
import torch
from torch import nn

import quant.ext
from quant.decode_validation import UNKNOWN, kernel_call_counter, rel_l2
from quant.precision import ieee_fp32_matmul
from quant.quantized_linear import QuantizedLinear

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

DTYPES = [torch.float16, torch.bfloat16]
REL_L2 = {torch.float16: 1e-3, torch.bfloat16: 1e-2}
K, N = 256, 96


@pytest.fixture(autouse=True)
def ieee_oracle():
    with ieee_fp32_matmul():
        yield


@pytest.fixture(params=DTYPES, ids=str)
def modules(request):
    """A CUDA-backend module and a reference module quantized from the same linear (with a bias, which
    Qwen3 lacks, so the bias path is exercised)."""
    torch.manual_seed(0)
    dtype = request.param
    linear = nn.Linear(K, N, bias=True, dtype=torch.bfloat16, device="cuda")
    cuda = QuantizedLinear.from_linear(linear, backend="cuda").to(dtype)
    reference = QuantizedLinear.from_linear(linear, backend="reference").to(dtype)
    return cuda, reference, dtype


def _x(shape, dtype):
    return torch.randn(shape, device="cuda").to(dtype)


@pytest.mark.parametrize("shape", [(K,), (1, K), (1, 1, K)], ids=str)
@torch.no_grad()
def test_single_vector_calls_the_kernel_once(modules, shape):
    cuda, reference, dtype = modules
    x = _x(shape, dtype)
    with kernel_call_counter(cuda) as counts:
        y = cuda(x)
    assert counts == {"": 1} and UNKNOWN not in counts
    assert y.shape == shape[:-1] + (N,) and y.dtype == dtype and y.is_contiguous()
    assert rel_l2(y, reference(x)) < REL_L2[dtype]


@torch.no_grad()
def test_non_contiguous_single_vector(modules):
    cuda, reference, dtype = modules
    x = _x((1, 2 * K), dtype)[:, ::2]
    assert not x.is_contiguous()
    with kernel_call_counter(cuda) as counts:
        y = cuda(x)
    assert counts == {"": 1}
    assert rel_l2(y, reference(x)) < REL_L2[dtype]


@pytest.mark.parametrize("shape", [(2, K), (1, 5, K), (2, 1, K)], ids=str)
@torch.no_grad()
def test_multi_vector_runs_the_reference_arithmetic(modules, shape):
    cuda, reference, dtype = modules
    x = _x(shape, dtype)
    with kernel_call_counter(cuda) as counts:
        y = cuda(x)
    assert sum(counts.values()) == 0
    assert torch.equal(y, reference(x))  # the same expression on the same inputs


@pytest.mark.parametrize("shape", [(0, K), (2, 0, K)], ids=str)
@torch.no_grad()
def test_empty_input_launches_nothing(modules, shape):
    cuda, _, dtype = modules
    with kernel_call_counter(cuda) as counts:
        y = cuda(_x(shape, dtype))
    assert y.shape == shape[:-1] + (N,) and y.dtype == dtype and sum(counts.values()) == 0


def test_fp32_activations_rejected(modules):
    cuda, _, _ = modules
    with torch.no_grad(), pytest.raises(RuntimeError, match="float16 or bfloat16"):
        cuda(torch.randn(1, K, device="cuda"))


def test_inference_only_for_every_vector_count(modules):
    cuda, _, dtype = modules
    for shape in [(1, K), (4, K)]:
        x = _x(shape, dtype).requires_grad_(True)
        with pytest.raises(RuntimeError, match="inference-only"):
            cuda(x)
        with torch.no_grad():
            cuda(x)
        cuda(x.detach())  # grad mode on, input without grad


def test_extension_loads_only_on_the_kernel_path(modules, monkeypatch):
    cuda, reference, dtype = modules

    def unavailable():
        raise RuntimeError("extension unavailable")

    monkeypatch.setattr(quant.ext, "gemv_ext", unavailable)
    x = _x((3, K), dtype)
    with torch.no_grad():
        assert torch.equal(cuda(x), reference(x))
        with pytest.raises(RuntimeError, match="extension unavailable"):
            cuda(x[:1])


@torch.no_grad()
def test_forward_hook_sees_the_returned_tensor(modules):
    cuda, _, dtype = modules
    seen = []
    handle = cuda.register_forward_hook(lambda module, args, out: seen.append(out))
    y = cuda(_x((1, K), dtype))
    handle.remove()
    assert len(seen) == 1 and seen[0] is y


@torch.no_grad()
def test_pre_hook_modification_is_honored(modules):
    cuda, reference, dtype = modules

    def zero_first_head(module, args):
        x = args[0].clone()
        x[..., :32] = 0
        return (x,)

    x = _x((1, K), dtype)
    handle = cuda.register_forward_pre_hook(zero_first_head)
    y = cuda(x)
    handle.remove()
    masked = x.clone()
    masked[..., :32] = 0
    assert rel_l2(y, reference(masked)) < REL_L2[dtype]
    assert rel_l2(y, reference(x)) > 10 * REL_L2[dtype]  # the mask had an effect


@torch.no_grad()
def test_buffers_and_state_dict_round_trip(modules):
    cuda, _, dtype = modules
    assert set(cuda.state_dict()) == {"qweight", "scales", "bias"}
    assert cuda.qweight.dtype == torch.int8 and cuda.scales.dtype == torch.float32
    restored = QuantizedLinear(K, N, bias=True, backend="cuda", device="cuda", bias_dtype=dtype)
    restored.load_state_dict(cuda.state_dict())
    x = _x((1, K), dtype)
    assert torch.equal(restored(x), cuda(x))
