import pytest
import torch
from torch import nn

from quant.quantized_linear import QuantizedLinear


@pytest.fixture
def linear():
    torch.manual_seed(0)
    return nn.Linear(256, 96, bias=False)


def test_matches_fp32_formula(linear):
    module = QuantizedLinear.from_linear(linear)
    x = torch.randn(5, 256)
    expected = (x @ module.qweight.float().T) * module.scales
    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)


def test_close_to_original_within_quantization_bound(linear):
    module = QuantizedLinear.from_linear(linear)
    x = torch.randn(5, 256)
    # |sum_k (w_hat - w) x_k| <= (s_n / 2) * sum_k |x_k|
    bound = module.scales * x.abs().sum(dim=-1, keepdim=True) / 2
    assert ((module(x) - linear(x)).abs() <= bound * (1 + 1e-4) + 1e-6).all()


def test_leading_shapes_and_dtype(linear):
    module = QuantizedLinear.from_linear(linear.to(torch.bfloat16))
    for shape in [(7, 256), (2, 7, 256), (1, 1, 256)]:
        x = torch.randn(shape, dtype=torch.bfloat16)
        y = module(x)
        assert y.shape == shape[:-1] + (96,)
        assert y.dtype == torch.bfloat16


def test_bias():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True)
    module = QuantizedLinear.from_linear(linear)
    x = torch.randn(3, 64)
    expected = (x @ module.qweight.float().T) * module.scales + linear.bias
    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)


def test_state_dict_round_trip(linear):
    module = QuantizedLinear.from_linear(linear)
    restored = QuantizedLinear(256, 96)
    restored.load_state_dict(module.state_dict())
    x = torch.randn(4, 256)
    assert torch.equal(restored(x), module(x))


@pytest.mark.parametrize("cast", [lambda m: m.half(), lambda m: m.to(torch.bfloat16), lambda m: m.double()])
def test_scales_stay_fp32_under_dtype_casts(linear, cast):
    module = QuantizedLinear.from_linear(linear)
    scales = module.scales.clone()
    cast(module)
    assert module.scales.dtype == torch.float32
    assert torch.equal(module.scales, scales)
    assert module.qweight.dtype == torch.int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scales_follow_device_moves(linear):
    module = QuantizedLinear.from_linear(linear).to("cuda", torch.float16)
    assert module.scales.device.type == "cuda" and module.scales.dtype == torch.float32
    assert module.qweight.device.type == "cuda"


def test_forward_hook_sees_returned_tensor(linear):
    module = QuantizedLinear.from_linear(linear)
    seen = []
    module.register_forward_hook(lambda mod, args, out: seen.append(out))
    y = module(torch.randn(2, 256))
    assert len(seen) == 1 and seen[0] is y


def test_cuda_backend_not_yet_implemented(linear):
    module = QuantizedLinear.from_linear(linear, backend="cuda")
    with pytest.raises(NotImplementedError):
        module(torch.randn(1, 256))


def test_unknown_backend_rejected(linear):
    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(linear, backend="triton")
