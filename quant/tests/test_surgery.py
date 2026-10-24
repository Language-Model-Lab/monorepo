import pytest
import torch
from torch import nn

from quant.quantized_linear import QuantizedLinear
from quant.surgery import quantize_model_

LINEARS_PER_LAYER = 7


def _input_ids(model):
    torch.manual_seed(1)
    return torch.randint(0, model.config.vocab_size, (2, 16))


def test_replaces_every_decoder_linear(tiny_model):
    replaced = quantize_model_(tiny_model)
    n_layers = tiny_model.config.num_hidden_layers
    assert len(replaced) == n_layers * LINEARS_PER_LAYER
    for path in replaced:
        assert isinstance(tiny_model.get_submodule(path), QuantizedLinear)
    leftover = [
        name for name, m in tiny_model.model.layers.named_modules() if isinstance(m, nn.Linear)
    ]
    assert leftover == []


def test_module_paths_unchanged(tiny_model):
    before = {name for name, _ in tiny_model.named_modules()}
    quantize_model_(tiny_model)
    assert {name for name, _ in tiny_model.named_modules()} == before


def test_lm_head_untouched_and_tied(tiny_model):
    lm_head = tiny_model.lm_head
    quantize_model_(tiny_model)
    assert tiny_model.lm_head is lm_head and type(lm_head) is nn.Linear
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.model.embed_tokens.weight.data_ptr()


def test_logits_close_to_unquantized(tiny_model):
    ids = _input_ids(tiny_model)
    with torch.no_grad():
        ref = tiny_model(ids).logits.float()
        quantize_model_(tiny_model)
        out = tiny_model(ids).logits.float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.05


def test_generate_runs(tiny_model):
    quantize_model_(tiny_model)
    ids = _input_ids(tiny_model)
    out = tiny_model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=0,
    )
    assert out.shape == (2, 16 + 4)


def test_predicate_restricts_targets(tiny_model):
    replaced = quantize_model_(tiny_model, predicate=lambda path: ".mlp." in path)
    assert len(replaced) == tiny_model.config.num_hidden_layers * 3
    assert isinstance(tiny_model.model.layers[0].self_attn.q_proj, nn.Linear)


def test_refuses_fp16_without_changing_model(tiny_model):
    tiny_model.half()
    with pytest.raises(ValueError, match="FP16"):
        quantize_model_(tiny_model)
    assert not any(isinstance(m, QuantizedLinear) for m in tiny_model.modules())


def test_cast_after_surgery_keeps_fp32_scales(tiny_model):
    quantize_model_(tiny_model)
    tiny_model.half()
    q_proj = tiny_model.model.layers[0].self_attn.q_proj
    assert q_proj.scales.dtype == torch.float32
    with torch.no_grad():
        assert tiny_model(_input_ids(tiny_model)).logits.dtype == torch.float16
