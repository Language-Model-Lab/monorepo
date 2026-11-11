import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

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


def test_quantized_vs_quantized_covers_every_hook_point(tiny_model):
    # K-001: two quantized models must keep their linears as hook points.
    model_a = _quantized_copy(tiny_model)
    result = compare_activations(model_a, copy.deepcopy(model_a), _batches(tiny_model))
    n_layers = tiny_model.config.num_hidden_layers
    assert len(result["compounded"]) == n_layers * (N_LINEARS + N_NORMS + 1)
    assert len(result["local"]) == n_layers * (N_LINEARS + N_NORMS)
    for mode in ("compounded", "local"):
        for point, regions in result[mode].items():
            for region in ("pos0", "rest"):
                assert regions[region]["rel_l2"] == 0 and regions[region]["max_abs_err"] == 0, (mode, point)


def test_perturbed_quantized_linear_is_detected(tiny_model):
    model_a = _quantized_copy(tiny_model)
    model_b = copy.deepcopy(model_a)
    # down_proj has no derived norm point, so only its own local point may change.
    target = "model.layers.1.mlp.down_proj"
    model_b.get_submodule(target).scales.mul_(1.01)
    result = compare_activations(model_a, model_b, _batches(tiny_model))
    for point, regions in result["local"].items():
        for region in ("pos0", "rest"):
            if point == target:
                assert regions[region]["rel_l2"] > 0, region
            else:
                assert regions[region]["rel_l2"] == 0, (point, region)


@pytest.mark.parametrize("mutation", ["non_linear_module", "missing_layer", "feature_dims"])
def test_mismatched_inventory_rejected(tiny_model, mutation):
    model_q = _quantized_copy(tiny_model)
    mlp = model_q.model.layers[0].mlp
    if mutation == "non_linear_module":
        mlp.down_proj = nn.Identity()
    elif mutation == "missing_layer":
        del model_q.model.layers[1]
    else:
        config = tiny_model.config
        mlp.down_proj = nn.Linear(config.intermediate_size, 2 * config.hidden_size, bias=False)
    with pytest.raises(ValueError, match="inventories differ"):
        compare_activations(tiny_model, model_q, _batches(tiny_model))


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.parametrize("failing_pass", ["reference", "quantized"])
def test_failure_removes_only_comparator_hooks(tiny_model, failing_pass):
    model_q = _quantized_copy(tiny_model)
    user_handles = [m.model.layers[0].mlp.register_forward_hook(lambda mod, args, out: None)
                    for m in (tiny_model, model_q)]

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    # model_q's down_proj runs inside the reference pass (local comparison); its act_fn only in its own pass.
    layer = model_q.model.layers[1].mlp
    if failing_pass == "reference":
        layer.down_proj.forward = boom
    else:
        layer.act_fn.forward = boom
    with pytest.raises(RuntimeError, match="boom"):
        compare_activations(tiny_model, model_q, _batches(tiny_model))
    assert _hook_count(tiny_model) == 1 and _hook_count(model_q) == 1
    for handle in user_handles:
        handle.remove()


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


def test_sliding_window_multiwindow_matches_per_token_reference(tiny_model):
    torch.manual_seed(6)
    n, ctx, stride = 20, 8, 5  # windows [0,8) [5,13) [10,18) [15,20): the last one is partial
    ids = torch.randint(0, tiny_model.config.vocab_size, (n,))
    result = sliding_window_eval({"m": tiny_model}, ids, ctx=ctx, stride=stride)
    assert result["scored"] == n - 1 and result["windows"] == 4

    # Independent reference: token j is scored once, by the first window [begin, end) that contains it,
    # from the full logits of that window at position j - begin - 1.
    nll, prev_end = 0.0, 0
    with torch.no_grad():
        for begin in range(0, n, stride):
            end = min(begin + ctx, n)
            logits = tiny_model(ids[begin:end][None], use_cache=False).logits[0].float()
            for j in range(max(prev_end, 1), end):
                nll -= logits[j - begin - 1].log_softmax(-1)[ids[j]].item()
            prev_end = end
            if end == n:
                break
    assert math.isclose(result["models"]["m"]["perplexity"], math.exp(nll / (n - 1)), rel_tol=1e-5)


@pytest.mark.parametrize("ctx, stride, match", [
    (8, 8, "stride"),  # K-004 reproduction: scored 21 of 23 tokens before the repair
    (8, 0, "stride"),
    (8, 9, "stride"),
    (1, 1, "ctx must be at least 2"),
])
def test_sliding_window_rejects_invalid_windows(tiny_model, ctx, stride, match):
    ids = torch.randint(0, tiny_model.config.vocab_size, (24,))
    with pytest.raises(ValueError, match=match):
        sliding_window_eval({"m": tiny_model}, ids, ctx=ctx, stride=stride)


def test_sliding_window_rejects_invalid_inputs(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (24,))
    window = {"ctx": 8, "stride": 4}
    with pytest.raises(ValueError, match="at least 2 tokens"):
        sliding_window_eval({"m": tiny_model}, ids[:1], **window)
    with pytest.raises(ValueError, match="at least 2 tokens"):
        sliding_window_eval({"m": tiny_model}, ids, max_tokens=1, **window)
    with pytest.raises(ValueError, match="1-D"):
        sliding_window_eval({"m": tiny_model}, ids[None], **window)
    with pytest.raises(ValueError, match="empty"):
        sliding_window_eval({}, ids, **window)
    with pytest.raises(ValueError, match="reference"):
        sliding_window_eval({"m": tiny_model}, ids, reference="other", **window)


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
