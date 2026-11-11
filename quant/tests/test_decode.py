"""The decode helpers are validated independently of the backends they later compare (K2-006)."""

from types import SimpleNamespace

import pytest
import torch

from quant.decode import cached_decode_logits, full_prefix_logits, greedy_continuation
from quant.surgery import quantize_model_

P = 7  # prompt length


@pytest.fixture
def fp32_model(tiny_model):
    """The tiny Qwen3, quantized (reference backend) and run in FP32 so the comparisons can be tight."""
    quantize_model_(tiny_model)
    return tiny_model.float()


def _ids(model, n, seed):
    return torch.randint(0, model.config.vocab_size, (n,), generator=torch.Generator().manual_seed(seed))


def _matches(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def _direct_rows(model, prompt, continuation, offset=0):
    """Logits at query positions P-1+offset .. P+L-2+offset, sliced here from one plain forward pass."""
    ids = torch.cat([prompt, continuation])[None]
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    start = prompt.numel() - 1 + offset
    return logits[start : start + continuation.numel()]


@pytest.mark.parametrize("length", [1, 2, 6])
def test_cached_rows_match_independently_sliced_logits(fp32_model, length):
    prompt, continuation = _ids(fp32_model, P, 0), _ids(fp32_model, length, 1)
    cached = cached_decode_logits(fp32_model, prompt, continuation)
    assert cached.shape == (length, fp32_model.config.vocab_size) and cached.dtype == torch.float32
    direct = _direct_rows(fp32_model, prompt, continuation)
    assert _matches(cached, direct)
    # Not bitwise: the direct pass is one token longer, which changes the matmul shapes.
    assert _matches(full_prefix_logits(fp32_model, prompt, continuation), direct)


def test_alignment_comparison_detects_shifts_and_wrong_tokens(fp32_model):
    prompt, continuation = _ids(fp32_model, P, 0), _ids(fp32_model, 6, 1)
    cached = cached_decode_logits(fp32_model, prompt, continuation)
    # Slicing one row early or one row late.
    assert not _matches(cached, _direct_rows(fp32_model, prompt, continuation, offset=-1))
    assert not _matches(cached, _direct_rows(fp32_model, prompt, continuation, offset=+1))
    # Feeding continuation[i] instead of continuation[i - 1] at decode step i.
    fed_wrong = torch.cat([continuation[1:], continuation[-1:]])
    wrong = cached_decode_logits(fp32_model, prompt, fed_wrong)
    assert not _matches(wrong[1:], _direct_rows(fp32_model, prompt, continuation)[1:])


class RecordingModel:
    """A deterministic stand-in that records every call. Logits encode the absolute position and the input
    token at that position, so rows can be checked against their expected query positions exactly."""

    vocab = 5

    def __init__(self):
        self.calls = []
        self.device = torch.device("cpu")

    def __call__(self, input_ids, past_key_values=None, use_cache=None, logits_to_keep=0, **kwargs):
        start = 0 if past_key_values is None else past_key_values.length
        cache = SimpleNamespace(length=start + input_ids.shape[1])
        self.calls.append({"input_ids": input_ids.clone(), "past": past_key_values, "cache": cache,
                           "use_cache": use_cache})
        positions = torch.arange(start, start + input_ids.shape[1], dtype=torch.float32)
        logits = positions[:, None] * 1000 + input_ids[0, :, None].float() + torch.arange(self.vocab) / 10
        if logits_to_keep:
            logits = logits[-logits_to_keep:]
        return SimpleNamespace(logits=logits[None], past_key_values=cache)

    @classmethod
    def expected_row(cls, position, token):
        return position * 1000 + token + torch.arange(cls.vocab, dtype=torch.float32) / 10


@pytest.mark.parametrize("length", [1, 2, 5])
def test_cached_decode_feeds_each_token_once_through_the_cache(length):
    model = RecordingModel()
    prompt, continuation = torch.arange(10, 10 + P), torch.arange(50, 50 + length)
    rows = cached_decode_logits(model, prompt, continuation)

    assert len(model.calls) == length
    first = model.calls[0]
    assert torch.equal(first["input_ids"], prompt[None]) and first["past"] is None and first["use_cache"] is True
    for i in range(1, length):
        call = model.calls[i]
        assert torch.equal(call["input_ids"], continuation[i - 1].view(1, 1))
        assert call["past"] is model.calls[i - 1]["cache"] and call["use_cache"] is True
    # Row i is query position P - 1 + i, whose input token is the prompt's last token, then continuation[i-1].
    inputs = torch.cat([prompt[-1:], continuation[:-1]])
    for i in range(length):
        assert torch.equal(rows[i], RecordingModel.expected_row(P - 1 + i, int(inputs[i])))


def test_full_prefix_runs_one_uncached_pass_over_the_same_positions():
    model = RecordingModel()
    prompt, continuation = torch.arange(10, 10 + P), torch.arange(50, 54)
    rows = full_prefix_logits(model, prompt, continuation)
    assert len(model.calls) == 1 and model.calls[0]["use_cache"] is False
    assert torch.equal(model.calls[0]["input_ids"], torch.cat([prompt, continuation[:-1]])[None])
    assert torch.equal(rows, cached_decode_logits(RecordingModel(), prompt, continuation))


def test_greedy_continuation_is_the_argmax_chain(fp32_model):
    prompt = _ids(fp32_model, P, 2)
    tokens = greedy_continuation(fp32_model, prompt, 5)
    assert tokens.shape == (5,) and tokens.dtype == torch.int64
    assert torch.equal(_direct_rows(fp32_model, prompt, tokens).argmax(-1), tokens)


def test_input_contracts():
    model = RecordingModel()
    prompt, continuation = torch.arange(P), torch.arange(3)
    for fn in (cached_decode_logits, full_prefix_logits):
        with pytest.raises(ValueError, match="empty"):
            fn(model, prompt[:0], continuation)
        with pytest.raises(ValueError, match="empty"):
            fn(model, prompt, continuation[:0])
        with pytest.raises(ValueError, match="1-D"):
            fn(model, prompt[None], continuation)
        with pytest.raises(ValueError, match="int64"):
            fn(model, prompt.float(), continuation)
    with pytest.raises(ValueError, match="at least 1"):
        greedy_continuation(model, prompt, 0)
