"""Real-model gate, fast version: 2 prompts, 8 tokens, FP16 (the full run is scripts/validate_cuda_decode.py)."""

import copy

import pytest
import torch

from quant.data import chat_inputs, load_prompts
from quant.decode_validation import GATES, run_decode_validation
from quant.loading import load_int8_model
from quant.surgery import set_backend_

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.fixture(scope="module")
def models():
    model_ref, tokenizer = load_int8_model(torch.float16, backend="reference")
    model_cuda = copy.deepcopy(model_ref)
    assert set_backend_(model_cuda, "cuda") == 28 * 7
    return model_ref, model_cuda, tokenizer


def test_real_model_cuda_decode_meets_every_gate(models):
    model_ref, model_cuda, tokenizer = models
    prompts = [chat_inputs(tokenizer, p["text"], enable_thinking=False)["input_ids"][0] for p in load_prompts()[:2]]
    report = run_decode_validation(model_ref, model_cuda, prompts, tokens=8, gate=GATES[torch.float16])
    assert report["passed"], report["checks"]
    assert report["quantized_linear_paths"] == 196
    for prompt in report["prompts"]:
        coverage = prompt["coverage"]["cuda_cached"]
        assert coverage["passed"] and coverage["observed_paths"] == 196 and coverage["calls_per_path"] == 7
        assert prompt["local"]["passed"] and prompt["local"]["calls"] == 196 * 7
