"""Correctness suite for the v0 GEMV kernels in csrc/gemv.cu (K-PLAN-002 stage 2).

Every case is checked against an FP64 oracle with a conservative FP32 forward-error envelope. Cases that
aggregate many rows are also checked against the FP32 reference expression by relative L2. The numerical
contract covers finite inputs whose unscaled FP32 accumulator stays finite and whose outputs are
representable; generators assert that they stay inside it.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from quant.gemv import gemv_16bit_control, int8_gemv
from quant.precision import ieee_fp32_matmul

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

DTYPES = [torch.float16, torch.bfloat16]
REL_L2 = {torch.float16: 1e-3, torch.bfloat16: 1e-2}
MODEL_SHAPES = [(2048, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (1024, 3072)]  # (N, K)
# K values off every power-of-two boundary; N values that leave a partial 128-thread block.
AWKWARD_SHAPES = [(1, 1), (127, 31), (129, 33), (255, 511), (1, 513), (129, 3073)]
SHAPES = MODEL_SHAPES + AWKWARD_SHAPES
SEEDS = [0, 1, 42]
MIN_ROWS_FOR_REL_L2 = 127  # with fewer rows relative L2 is a single, possibly ill-conditioned, element
U = 2.0**-24  # FP32 unit roundoff
ACC_LIMIT = 1e36  # bound on the unscaled FP32 accumulator, far below the FP32 maximum (3.4e38)
ADVERSARIAL = ["signed_extrema", "alternating_cancellation", "catastrophic_cancellation", "small_scales",
               "large_activations"]


@pytest.fixture(autouse=True)
def ieee_oracle():
    with ieee_fp32_matmul():
        yield


def _shape_id(shape):
    return f"{shape[0]}x{shape[1]}"


def _cuda(*tensors):
    return tuple(t.cuda() if t is not None else None for t in tensors)


def _gamma(m: int) -> float:
    return m * U / (1 - m * U)


def fp64_terms(x, w, scales, bias):
    """FP64 exact output [N] and the sum of absolute terms that bounds its FP32 rounding error."""
    products = w.double().cpu() * x.double().cpu()
    if scales is not None:
        products = products * scales.double().cpu()[:, None]
    exact, magnitude = products.sum(1), products.abs().sum(1)
    if bias is not None:
        exact, magnitude = exact + bias.double().cpu(), magnitude + bias.double().cpu().abs()
    return exact, magnitude


def assert_in_domain(x, w, scales, bias):
    unscaled = x.shape[-1] * w.float().abs().max().item() * x.float().abs().max().item()
    assert unscaled < ACC_LIMIT, f"unscaled accumulator bound {unscaled:.3e} leaves the numerical domain"
    exact, _ = fp64_terms(x, w, scales, bias)
    assert exact.abs().max().item() < torch.finfo(x.dtype).max / 2, "output not representable"


def assert_within_envelope(y, x, w, scales, bias):
    """The kernel's output must lie between the FP64 result +- gamma(2K+4) * sum|terms|, both endpoints
    rounded to the output dtype (rounding is monotone, so a correctly bounded FP32 value stays inside)."""
    exact, magnitude = fp64_terms(x, w, scales, bias)
    bound = _gamma(2 * x.shape[-1] + 4) * magnitude
    lo, hi = (exact - bound).to(y.dtype).double(), (exact + bound).to(y.dtype).double()
    out = y.double().cpu()[0]
    outside = ~((out >= lo) & (out <= hi))
    assert not outside.any(), (
        f"{int(outside.sum())} of {out.numel()} outputs outside the FP32 envelope; "
        f"first at row {int(outside.nonzero()[0])}: {out[outside][0].item()} not in "
        f"[{lo[outside][0].item()}, {hi[outside][0].item()}]"
    )


def rel_l2(test, ref) -> float:
    return ((test.float() - ref).norm() / ref.norm()).item()


def int8_reference(x, q, scales, bias):
    # QuantizedLinear's reference expression, before the final cast.
    y = F.linear(x.float(), q.float()) * scales
    return y + bias if bias is not None else y


def control_reference(x, w, bias):
    y = F.linear(x.float(), w.float())
    return y + bias if bias is not None else y


def int8_case(n, k, dtype, seed, bias):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    # Log-uniform over the range of real Qwen3-0.6B row scales (findings section 4).
    scales = torch.empty(n).uniform_(math.log(4e-5), math.log(1e-2), generator=g).exp()
    x = torch.randn(1, k, generator=g).to(dtype)
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return _cuda(x, q, scales, b)


def control_case(n, k, dtype, seed, bias):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).to(dtype)
    x = torch.randn(1, k, generator=g).to(dtype)
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return _cuda(x, w, b)


def adversarial_case(name, kind, dtype):
    """Small inputs that stress rounding. For the control kernel the integer weights are stored exactly in
    the 16-bit dtype and there is no scale."""
    n, k = 8, 64
    g = torch.Generator().manual_seed(7)
    scales, bias = torch.full((n,), 1e-2), None
    if name == "signed_extrema":
        x = torch.randn(1, k, generator=g)
        w = 127.0 * (torch.randint(0, 2, (n, k), generator=g) * 2 - 1)
    elif name == "alternating_cancellation":
        # +127, -127, ... against a constant activation: every pair cancels exactly, leaving the bias.
        x = torch.full((1, k), 3.0)
        w = torch.tensor([127.0, -127.0]).repeat(n, k // 2)
        bias = torch.linspace(-1, 1, n)
    elif name == "catastrophic_cancellation":
        x = torch.zeros(1, k)
        x[0, :3] = torch.tensor([1000.0, -1000.0, 0.5])
        w = torch.zeros(n, k)
        w[:, 0], w[:, 1], w[:, 2] = 127.0, 127.0, torch.arange(n) - 4.0
    elif name == "small_scales":
        x = torch.randn(1, k, generator=g)
        w = torch.randint(-127, 128, (n, k), generator=g).float()
        scales = torch.logspace(math.log10(4e-5), -2, n)
        bias = torch.randn(n, generator=g)
    elif name == "large_activations":
        # Large but inside the domain: the unscaled accumulator, not only the scaled result, is bounded.
        big = 1e4 if dtype == torch.float16 else 1e30
        x = (torch.rand(1, k, generator=g) * 2 - 1) * big
        if kind == "int8":
            w = torch.randint(-127, 128, (n, k), generator=g).float()
            scales = torch.full((n,), 4e-5)
        else:
            w = (torch.rand(n, k, generator=g) * 2 - 1) * (1e-3 if dtype == torch.float16 else 1.0)
    else:
        raise ValueError(name)
    w = w.to(torch.int8) if kind == "int8" else w.to(dtype)
    return _cuda(x.to(dtype), w, scales if kind == "int8" else None, bias)


@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_int8_gemv_matches_references(shape, seed, dtype, bias):
    n, k = shape
    x, q, scales, b = int8_case(n, k, dtype, seed, bias)
    assert_in_domain(x, q, scales, b)
    y = int8_gemv(x, q, scales, b)
    assert y.shape == (1, n) and y.dtype == dtype and y.device == x.device and y.is_contiguous()
    assert_within_envelope(y, x, q, scales, b)
    if n >= MIN_ROWS_FOR_REL_L2:
        assert rel_l2(y, int8_reference(x, q, scales, b)) < REL_L2[dtype]


@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_control_matches_references(shape, dtype, bias):
    n, k = shape
    x, w, b = control_case(n, k, dtype, 0, bias)
    assert_in_domain(x, w, None, b)
    y = gemv_16bit_control(x, w, b)
    assert y.shape == (1, n) and y.dtype == dtype and y.device == x.device and y.is_contiguous()
    assert_within_envelope(y, x, w, None, b)
    if n >= MIN_ROWS_FOR_REL_L2:
        assert rel_l2(y, control_reference(x, w, b)) < REL_L2[dtype]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("kind", ["int8", "control"])
@pytest.mark.parametrize("name", ADVERSARIAL)
def test_adversarial_inputs_within_fp32_envelope(name, kind, dtype):
    x, w, scales, b = adversarial_case(name, kind, dtype)
    assert_in_domain(x, w, scales, b)
    y = int8_gemv(x, w, scales, b) if kind == "int8" else gemv_16bit_control(x, w, b)
    assert_within_envelope(y, x, w, scales, b)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_zero_input_gives_exact_zero_or_bias(dtype):
    n, k = 129, 33
    x = torch.zeros(1, k, dtype=dtype, device="cuda")
    q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    scales = torch.rand(n, device="cuda") * 1e-2
    w = torch.randn(n, k, device="cuda").to(dtype)
    bias = torch.randn(n, device="cuda")
    zeros = torch.zeros(1, n, dtype=dtype, device="cuda")
    assert torch.equal(int8_gemv(x, q, scales), zeros)
    assert torch.equal(int8_gemv(x, q, scales, bias), bias.to(dtype)[None])
    assert torch.equal(gemv_16bit_control(x, w), zeros)
    assert torch.equal(gemv_16bit_control(x, w, bias), bias.to(dtype)[None])


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_storage_offset_views(dtype):
    # Contiguous views that do not start on a 16-byte boundary: v0 must not assume alignment.
    n, k = 129, 513
    x = torch.randn(1, k + 3, device="cuda").to(dtype)[:, 3:]
    q = torch.randint(-127, 128, (n + 1, k), dtype=torch.int8, device="cuda")[1:]
    scales = (torch.rand(n + 1, device="cuda") * 1e-2)[1:]
    bias = torch.randn(n + 1, device="cuda")[1:]
    w = torch.randn(n + 1, k, device="cuda").to(dtype)[1:]
    assert x.is_contiguous() and q.is_contiguous() and x.data_ptr() % 16 and q.data_ptr() % 16
    assert_within_envelope(int8_gemv(x, q, scales, bias), x, q, scales, bias)
    assert_within_envelope(gemv_16bit_control(x, w, bias), x, w, None, bias)


def _valid_int8(n=129, k=33):
    x = torch.randn(1, k, device="cuda").half()
    q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    scales = torch.rand(n, device="cuda") * 1e-2
    bias = torch.randn(n, device="cuda")
    return x, q, scales, bias


INVALID_INT8 = {
    "x_float32": lambda x, q, s, b: (x.float(), q, s, b),
    "x_on_cpu": lambda x, q, s, b: (x.cpu(), q, s, b),
    "x_two_rows": lambda x, q, s, b: (x.repeat(2, 1), q, s, b),
    "x_one_dim": lambda x, q, s, b: (x[0], q, s, b),
    "x_non_contiguous": lambda x, q, s, b: (torch.cat([x, x], 1)[:, ::2], q, s, b),
    "k_zero": lambda x, q, s, b: (x[:, :0], q[:, :0].contiguous(), s, b),
    "n_zero": lambda x, q, s, b: (x, q[:0], s[:0], b[:0]),
    "qweight_float": lambda x, q, s, b: (x, q.float(), s, b),
    "qweight_non_contiguous": lambda x, q, s, b: (x, q.t().contiguous().t(), s, b),
    "qweight_one_dim": lambda x, q, s, b: (x, q[0], s, b),
    "qweight_k_mismatch": lambda x, q, s, b: (x, q[:, :-1].contiguous(), s, b),
    "qweight_on_cpu": lambda x, q, s, b: (x, q.cpu(), s, b),
    "scales_float16": lambda x, q, s, b: (x, q, s.half(), b),
    "scales_wrong_length": lambda x, q, s, b: (x, q, s[:-1], b),
    "scales_two_dim": lambda x, q, s, b: (x, q, s[:, None], b),
    "scales_on_cpu": lambda x, q, s, b: (x, q, s.cpu(), b),
    "bias_float16": lambda x, q, s, b: (x, q, s, b.half()),
    "bias_wrong_length": lambda x, q, s, b: (x, q, s, b[:-1]),
    "bias_on_cpu": lambda x, q, s, b: (x, q, s, b.cpu()),
}

INVALID_CONTROL = {
    "weight_other_dtype": lambda x, w, b: (x, w.bfloat16(), b),
    "weight_int8": lambda x, w, b: (x, w.to(torch.int8), b),
    "weight_non_contiguous": lambda x, w, b: (x, w.t().contiguous().t(), b),
    "weight_k_mismatch": lambda x, w, b: (x, w[:, :-1].contiguous(), b),
    "x_float32": lambda x, w, b: (x.float(), w.float(), b),
    "bias_float16": lambda x, w, b: (x, w, b.half()),
}


@pytest.mark.parametrize("name", INVALID_INT8)
def test_int8_gemv_rejects_invalid_metadata(name):
    with pytest.raises(RuntimeError):
        int8_gemv(*INVALID_INT8[name](*_valid_int8()))


@pytest.mark.parametrize("name", INVALID_CONTROL)
def test_control_rejects_invalid_metadata(name):
    x, _, _, bias = _valid_int8()
    w = torch.randn(bias.numel(), x.shape[1], device="cuda").half()
    with pytest.raises(RuntimeError):
        gemv_16bit_control(*INVALID_CONTROL[name](x, w, bias))


def test_runs_on_the_current_stream():
    # The input is only written on a side stream, after a long sleep. A kernel launched on any other stream
    # would read the zeros that are there before the copy lands.
    x_true, q, scales, bias = _valid_int8()
    x = torch.zeros_like(x_true)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)
        x.copy_(x_true)
        y = int8_gemv(x, q, scales, bias)
    side.synchronize()
    assert_within_envelope(y, x_true, q, scales, bias)


def test_inputs_and_current_device_unchanged():
    tensors = _valid_int8()
    clones = [t.clone() for t in tensors]
    device = torch.cuda.current_device()
    int8_gemv(*tensors)
    gemv_16bit_control(tensors[0], torch.randn(129, 33, device="cuda").half())
    torch.cuda.synchronize()
    assert torch.cuda.current_device() == device
    assert all(torch.equal(t, c) for t, c in zip(tensors, clones))


def test_inference_only_grad_contract():
    x, q, scales, bias = _valid_int8()
    w = torch.randn(129, 33, device="cuda").half()
    x_grad = x.clone().requires_grad_(True)
    assert torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="inference-only"):
        int8_gemv(x_grad, q, scales, bias)
    with pytest.raises(RuntimeError, match="inference-only"):
        gemv_16bit_control(x_grad, w, bias)
    # Grad mode alone is fine, and so is a grad-requiring input under no_grad.
    int8_gemv(x, q, scales, bias)
    gemv_16bit_control(x, w, bias)
    with torch.no_grad():
        int8_gemv(x_grad, q, scales, bias)
        gemv_16bit_control(x_grad, w, bias)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs")
def test_mixed_devices_rejected():
    x, q, scales, bias = _valid_int8()
    with pytest.raises(RuntimeError, match="is on"):
        int8_gemv(x, q.to("cuda:1"), scales, bias)
    with pytest.raises(RuntimeError, match="is on"):
        gemv_16bit_control(x, torch.randn(129, 33, device="cuda:1").half(), bias)
