import pytest
import torch

from quant.loading import load_model
from quant.quantized_linear import QuantizedLinear
from quant.surgery import quantize_model_

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.fixture(scope="module")
def quantized():
    model, tokenizer = load_model(torch.bfloat16)
    replaced = quantize_model_(model)
    return model, tokenizer, replaced


def test_all_196_linears_replaced(quantized):
    model, _, replaced = quantized
    assert len(replaced) == 28 * 7
    assert all(isinstance(model.get_submodule(p), QuantizedLinear) for p in replaced)


def test_embedding_still_tied(quantized):
    model, _, _ = quantized
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_generates_text(quantized):
    model, tokenizer, _ = quantized
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    torch.manual_seed(0)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    assert "Paris" in text
