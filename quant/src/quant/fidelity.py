"""Fidelity ladder rungs 2-4 : compare a reference model with a modified copy.

Position 0 is reported separately from positions >= 1 throughout: Qwen3-0.6B writes a large activation
(thousands) into the residual stream at position 0, which would dominate any all-token average.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

LOGIT_SLICE = 256  # tokens per slice for the float32 log-softmax, to bound memory
SAMPLING_NON_THINKING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}  # model card
NORM_OF_PROJ = {"self_attn.q_proj": "self_attn.q_norm", "self_attn.k_proj": "self_attn.k_norm"}

#Running comparison of reference vs test activations, one row per token.
class ErrorStats:

    def __init__(self):
        self.tokens = 0
        self.cos_sum = 0.0
        self.cos_min = math.inf
        self.err_sq = 0.0
        self.ref_sq = 0.0
        self.max_abs_err = 0.0
        self.ref_absmax = 0.0
        self.test_absmax = 0.0

    def update(self, ref: torch.Tensor, test: torch.Tensor) -> None:
        if ref.shape[0] == 0:
            return
        ref, test = ref.float(), test.float()
        diff = test - ref
        cos = F.cosine_similarity(ref, test, dim=-1, eps=1e-12)
        values = torch.stack([
            cos.sum(), cos.min(), diff.pow(2).sum(), ref.pow(2).sum(),
            diff.abs().max(), ref.abs().max(), test.abs().max(),
        ]).tolist()
        self.tokens += ref.shape[0]
        self.cos_sum += values[0]
        self.cos_min = min(self.cos_min, values[1])
        self.err_sq += values[2]
        self.ref_sq += values[3]
        self.max_abs_err = max(self.max_abs_err, values[4])
        self.ref_absmax = max(self.ref_absmax, values[5])
        self.test_absmax = max(self.test_absmax, values[6])

    def result(self) -> dict:
        return {
            "tokens": self.tokens,
            "cos_mean": self.cos_sum / self.tokens,
            "cos_min": self.cos_min,
            "rel_l2": math.sqrt(self.err_sq / self.ref_sq) if self.ref_sq > 0 else 0.0,
            "max_abs_err": self.max_abs_err,
            "ref_absmax": self.ref_absmax,
            "test_absmax": self.test_absmax,
        }


class _PositionSplitStats:
    def __init__(self):
        self.pos0, self.rest = ErrorStats(), ErrorStats()

    def update(self, ref: torch.Tensor, test: torch.Tensor) -> None:
        # [B, T, ...] -> position 0 rows and position >= 1 rows of [tokens, features]
        ref = ref.reshape(ref.shape[0], ref.shape[1], -1)
        test = test.reshape(test.shape[0], test.shape[1], -1)
        self.pos0.update(ref[:, 0], test[:, 0])
        self.rest.update(ref[:, 1:].reshape(-1, ref.shape[-1]), test[:, 1:].reshape(-1, test.shape[-1]))

    def result(self) -> dict:
        return {"pos0": self.pos0.result(), "rest": self.rest.result()}


def _first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def hook_points(model) -> list[str]:
    """Per decoder layer: every linear, q_norm/k_norm if present, and the layer output (residual stream)."""
    points = []
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) or name in NORM_OF_PROJ.values():
                points.append(f"{prefix}.{name}")
        points.append(prefix)
    return points


@torch.no_grad()
def compare_activations(model_ref, model_q, batches: Iterable[torch.Tensor]) -> dict:
    """Rung 2. For each batch of input ids, run model_ref then model_q.

    compounded: model_q's activations against model_ref's, each on its own forward pass (errors accumulate
        through depth).
    local: each linear of model_q applied to the *reference* input of that linear, against the reference
        output (that layer's own error). For q_proj/k_proj the same comparison is repeated after q_norm/k_norm.
    Both models must share module paths (model_q is model_ref after surgery).
    """
    points = hook_points(model_ref)
    linears = [p for p in points if isinstance(model_ref.get_submodule(p), nn.Linear)]
    head_dim = model_ref.config.head_dim
    compounded = {p: _PositionSplitStats() for p in points}
    local = {}
    for p in linears:
        local[p] = _PositionSplitStats()
        suffix = p.split(".", 3)[3]
        if suffix in NORM_OF_PROJ:
            local[p.replace(suffix, NORM_OF_PROJ[suffix])] = _PositionSplitStats()
    local = {p: local[p] for p in points if p in local}  # forward order

    def store_hook(path, captured):
        def hook(module, args, output):
            captured[path] = _first_tensor(output)
        return hook

    def local_hook(path):
        q_module = model_q.get_submodule(path)
        suffix = path.split(".", 3)[3]
        # model_q's norm modules are unquantized and carry no hooks during the reference pass
        norm = model_q.get_submodule(path.replace(suffix, NORM_OF_PROJ[suffix])) if suffix in NORM_OF_PROJ else None

        def hook(module, args, output):
            test = q_module(args[0])
            local[path].update(output, test)
            if norm is not None:
                heads = output.shape[:-1] + (-1, head_dim)
                local[path.replace(suffix, NORM_OF_PROJ[suffix])].update(
                    norm(output.view(heads)), norm(test.view(heads))
                )
        return hook

    def compare_hook(path, captured):
        def hook(module, args, output):
            compounded[path].update(captured[path], _first_tensor(output))
        return hook

    for input_ids in batches:
        input_ids = input_ids.to(model_ref.device)
        captured = {}
        handles = [model_ref.get_submodule(p).register_forward_hook(store_hook(p, captured)) for p in points]
        handles += [model_ref.get_submodule(p).register_forward_hook(local_hook(p)) for p in linears]
        try:
            model_ref(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        handles = [model_q.get_submodule(p).register_forward_hook(compare_hook(p, captured)) for p in points]
        try:
            model_q(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        del captured

    return {
        "compounded": {p: s.result() for p, s in compounded.items()},
        "local": {p: s.result() for p, s in local.items()},
    }


@torch.no_grad()
def sliding_window_eval(
    models: dict[str, nn.Module],
    ids: torch.Tensor,
    ctx: int = 1024,
    stride: int = 512,
    max_tokens: int | None = None,
    reference: str | None = None,
) -> dict:
    """Rung 3. Sliding-window NLL over a 1-D token stream; every token after the first is scored once.

    All models run on the same windows. Each non-reference model is also compared with the reference on the
    scored positions: KL(reference || model), top-1 agreement, largest logit difference.
    """
    if max_tokens is not None:
        ids = ids[:max_tokens]
    reference = reference or next(iter(models))
    n = ids.numel()
    nll = dict.fromkeys(models, 0.0)
    others = [name for name in models if name != reference]
    vs = {name: {"kl_sum": 0.0, "kl_max": 0.0, "top1": 0, "max_abs_logit_diff": 0.0} for name in others}
    scored = windows = prev_end = 0

    for begin in range(0, n, stride):
        end = min(begin + ctx, n)
        new = end - prev_end  # tokens this window adds
        length = end - begin
        keep = min(length, new + 1)  # logits needed: positions predicting the new tokens, plus the last
        window = ids[begin:end][None]
        targets = window[0, length - keep + 1:]
        logits = {}
        for name, model in models.items():
            out = model(window.to(model.device), use_cache=False, logits_to_keep=keep).logits[0, :-1]
            logits[name] = out
        targets = targets.to(logits[reference].device)
        for s in range(0, targets.numel(), LOGIT_SLICE):
            e = min(s + LOGIT_SLICE, targets.numel())
            t = targets[s:e, None]
            logprobs = {name: lg[s:e].float().log_softmax(-1) for name, lg in logits.items()}
            for name, lp in logprobs.items():
                nll[name] -= lp.gather(1, t).sum().item()
            lp_ref = logprobs[reference]
            for name in others:
                kl = (lp_ref.exp() * (lp_ref - logprobs[name])).sum(-1)
                vs[name]["kl_sum"] += kl.sum().item()
                vs[name]["kl_max"] = max(vs[name]["kl_max"], kl.max().item())
                vs[name]["top1"] += (lp_ref.argmax(-1) == logprobs[name].argmax(-1)).sum().item()
                diff = (logits[reference][s:e].float() - logits[name][s:e].float()).abs().max().item()
                vs[name]["max_abs_logit_diff"] = max(vs[name]["max_abs_logit_diff"], diff)
        scored += targets.numel()
        windows += 1
        prev_end = end
        if end == n:
            break

    return {
        "ctx": ctx,
        "stride": stride,
        "tokens": n,
        "scored": scored,
        "windows": windows,
        "reference": reference,
        "models": {name: {"nll_sum": v, "perplexity": math.exp(v / scored)} for name, v in nll.items()},
        "vs_reference": {
            name: {
                "kl_mean": v["kl_sum"] / scored,
                "kl_max": v["kl_max"],
                "top1_agreement": v["top1"] / scored,
                "max_abs_logit_diff": v["max_abs_logit_diff"],
            }
            for name, v in vs.items()
        },
    }


def _first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def generate_side_by_side(
    models: dict[str, nn.Module],
    inputs_list: list[dict],
    seed: int = 0,
    max_new_tokens: int = 128,
    reference: str | None = None,
    **generate_kwargs,
) -> list[dict]:
    """Rung 4. Seeded sampling (model-card non-thinking settings by default) on each input, for every model.

    The seed is reset before every generate call, so models consume random numbers identically and outputs
    differ only where the logits differ enough to change a sample. Returns, per input, the generated token
    ids per model and the index of the first token where each model differs from the reference.
    """
    reference = reference or next(iter(models))
    kwargs = {**SAMPLING_NON_THINKING, **generate_kwargs}
    rows = []
    for i, inputs in enumerate(inputs_list):
        token_ids = {}
        for name, model in models.items():
            batch = {k: v.to(model.device) for k, v in inputs.items()}
            torch.manual_seed(seed + i)
            out = model.generate(**batch, max_new_tokens=max_new_tokens, **kwargs)
            token_ids[name] = out[0, batch["input_ids"].shape[-1]:].tolist()
        rows.append({
            "token_ids": token_ids,
            "first_divergence": {
                name: _first_difference(token_ids[reference], ids) for name, ids in token_ids.items() if name != reference
            },
        })
    return rows
