import pytest
import torch

from quant.quantize import (
    FP32_SLACK,
    QMAX,
    dequantize_int8_per_channel,
    quantize_int8_per_channel,
    weight_error_stats,
)


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(96, 256) * 0.02


def test_output_types_and_shapes(weight):
    q, scales = quantize_int8_per_channel(weight)
    assert q.dtype == torch.int8 and q.shape == weight.shape and q.is_contiguous()
    assert scales.dtype == torch.float32 and scales.shape == (weight.shape[0],)


def test_round_trip_error_within_half_step(weight):
    q, scales = quantize_int8_per_channel(weight)
    err = (dequantize_int8_per_channel(q, scales) - weight).abs()
    assert (err <= scales[:, None] * (0.5 + FP32_SLACK)).all()


def test_symmetric_range(weight):
    q, _ = quantize_int8_per_channel(weight)
    assert q.min() >= -QMAX and q.max() <= QMAX


def test_row_max_maps_to_full_range(weight):
    q, _ = quantize_int8_per_channel(weight)
    assert (q.abs().amax(dim=1) == QMAX).all()


def test_zero_row():
    w = torch.randn(4, 64)
    w[2] = 0
    q, scales = quantize_int8_per_channel(w)
    assert (q[2] == 0).all()
    assert torch.isfinite(scales).all() and scales[2] > 0
    assert (dequantize_int8_per_channel(q, scales)[2] == 0).all()


def test_per_channel_isolates_outlier_row(weight):
    w = weight.clone()
    w[0] *= 1000
    _, scales = quantize_int8_per_channel(w)
    _, scales_clean = quantize_int8_per_channel(weight)
    torch.testing.assert_close(scales[1:], scales_clean[1:], rtol=0, atol=0)


def test_bf16_matches_fp32_path(weight):
    w_bf16 = weight.to(torch.bfloat16)
    q_a, s_a = quantize_int8_per_channel(w_bf16)
    q_b, s_b = quantize_int8_per_channel(w_bf16.float())
    assert torch.equal(q_a, q_b) and torch.equal(s_a, s_b)


def test_refuses_fp16(weight):
    with pytest.raises(ValueError, match="FP16"):
        quantize_int8_per_channel(weight.half())


def test_refuses_non_2d():
    with pytest.raises(ValueError):
        quantize_int8_per_channel(torch.randn(2, 3, 4))


def test_deterministic(weight):
    q_a, s_a = quantize_int8_per_channel(weight)
    q_b, s_b = quantize_int8_per_channel(weight)
    assert torch.equal(q_a, q_b) and torch.equal(s_a, s_b)


def test_error_stats(weight):
    q, scales = quantize_int8_per_channel(weight)
    stats = weight_error_stats(weight, q, scales)
    assert stats["max_err_over_scale"] <= 0.5 + FP32_SLACK
    assert stats["clipped_frac"] == 0
    assert 0 < stats["rel_fro_err"] < 0.02
    assert stats["sqnr_db"] > 30
