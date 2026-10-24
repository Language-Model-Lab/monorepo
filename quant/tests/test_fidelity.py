import copy
import math

import torch
import torch.nn.functional as F

from quant.fidelity import compare_activations, generate_side_by_side, sliding_window_eval
from quant.surgery import quantize_model_

N_LINEARS, N_NORMS = 7, 2


def _batches(model, n=2, length=12):
    torch.manual_seed(1)
    return [torch.randint(0, model.config.vocab_size, (1, length)) for _ in range(n)]


def _quantized_copy(model):
    model_q = copy.deepcopy(model)
    quantize_model_(model_q)
    return model_q


def test_identical_models_have_zero_error(tiny_model):
    result = compare_activations(tiny_model, copy.deepcopy(tiny_model), _batches(tiny_model))
    for mode in ("compounded", "local"):
        for point, regions in result[mode].items():
            for region in ("pos0", "rest"):
                stats = regions[region]
                assert stats["rel_l2"] == 0 and stats["max_abs_err"] == 0, (mode, point, region)
                assert math.isclose(stats["cos_min"], 1, abs_tol=1e-5)


def test_quantized_copy_covers_every_hook_point(tiny_model):
    batches = _batches(tiny_model, n=3, length=10)
    result = compare_activations(tiny_model, _quantized_copy(tiny_model), batches)
    n_layers = tiny_model.config.num_hidden_layers
    assert len(result["compounded"]) == n_layers * (N_LINEARS + N_NORMS + 1)
    assert len(result["local"]) == n_layers * (N_LINEARS + N_NORMS)
    for mode in ("compounded", "local"):
        for regions in result[mode].values():
            assert regions["pos0"]["tokens"] == 3
            assert regions["rest"]["tokens"] == 3 * 9
            assert all(math.isfinite(v) for r in regions.values() for v in r.values())
    for point, regions in result["local"].items():
        assert 0 < regions["rest"]["rel_l2"] < 0.05, point


def test_local_norm_points_follow_their_projection(tiny_model):
    result = compare_activations(tiny_model, _quantized_copy(tiny_model), _batches(tiny_model))
    assert "model.layers.0.self_attn.q_norm" in result["local"]
    assert "model.layers.0.self_attn.k_norm" in result["local"]
    assert "model.layers.0" not in result["local"]


def test_sliding_window_scores_every_token_once(tiny_model):
    torch.manual_seed(2)
    ids = torch.randint(0, tiny_model.config.vocab_size, (100,))
    result = sliding_window_eval({"m": tiny_model}, ids, ctx=32, stride=16)
    assert result["scored"] == 99


def test_single_window_perplexity_matches_manual(tiny_model):
    torch.manual_seed(3)
    ids = torch.randint(0, tiny_model.config.vocab_size, (40,))
    result = sliding_window_eval({"m": tiny_model}, ids, ctx=64, stride=32)
    with torch.no_grad():
        logits = tiny_model(ids[None]).logits[0, :-1].float()
    expected = math.exp(F.cross_entropy(logits, ids[1:]).item())
    assert math.isclose(result["models"]["m"]["perplexity"], expected, rel_tol=1e-5)


def test_identical_models_agree_on_logits(tiny_model):
    torch.manual_seed(4)
    ids = torch.randint(0, tiny_model.config.vocab_size, (60,))
    result = sliding_window_eval({"a": tiny_model, "b": copy.deepcopy(tiny_model)}, ids, ctx=32, stride=16)
    vs = result["vs_reference"]["b"]
    assert vs["kl_max"] < 1e-6 and vs["top1_agreement"] == 1 and vs["max_abs_logit_diff"] == 0


def test_generate_side_by_side_identical_models(tiny_model):
    torch.manual_seed(5)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    inputs = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}]
    rows = generate_side_by_side(
        {"a": tiny_model, "b": copy.deepcopy(tiny_model)}, inputs, max_new_tokens=6, pad_token_id=0
    )
    assert rows[0]["token_ids"]["a"] == rows[0]["token_ids"]["b"]
    assert len(rows[0]["token_ids"]["a"]) == 6
    assert rows[0]["first_divergence"] == {"b": None}
