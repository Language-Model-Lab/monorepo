"""End to end decode timing: time to first token and decode tokens/s."""

import time

import torch

MARKER = "<<<FILL>>>"
SAMPLING_NON_THINKING = {"temperature": 0.7, "top_k": 20, "top_p": 0.8}  # model card

#A non-thinking single-turn chat prompt of exactly `length` tokens, filled with corpus tokens.
def build_prompt_ids(tokenizer, corpus_ids: torch.Tensor, length: int) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": MARKER}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    before, after = text.split(MARKER)
    pre = tokenizer(before, add_special_tokens=False).input_ids
    post = tokenizer(after, add_special_tokens=False).input_ids
    fill = length - len(pre) - len(post)
    if fill < 1:
        raise ValueError(f"length {length} is shorter than the chat template ({len(pre) + len(post)} tokens)")
    if corpus_ids.numel() < fill:
        raise ValueError(f"corpus has {corpus_ids.numel()} tokens, need {fill}")
    return torch.tensor(pre + corpus_ids[:fill].tolist() + post)


def sample_next(
    logits: torch.Tensor, generator: torch.Generator, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Temperature, then top-k, then top-p (Hugging Face's order), on logits [B, V]. Returns ids [B, 1].
    Runs entirely on the logits' device, so it does not synchronize."""
    values, indices = (logits.float() / temperature).topk(top_k, dim=-1)  # sorted, descending
    probs = values.softmax(-1)
    # Keep the smallest prefix whose mass reaches top_p: drop tokens whose preceding mass already does.
    probs = probs.masked_fill(probs.cumsum(-1) - probs >= top_p, 0.0)
    choice = torch.multinomial(probs, 1, generator=generator)
    return indices.gather(-1, choice)


"""
    Prefill the prompt, then decode gen_tokens -1 more tokens, ignoring end of sequence.

    TTFT: CUDA event time from the start of prefill to the first sampled token.
    Decode: CUDA event time of the remaining steps; decode tokens/s excludes the first token.
    Both include CPU side and kernel launch overhead, since the GPU idles while waiting for work.
    """
@torch.no_grad()
def decode_run(model, prompt_ids: torch.Tensor, gen_tokens: int, seed: int = 0, sampling: dict | None = None) -> dict:
    sampling = sampling or SAMPLING_NON_THINKING
    device = model.device
    ids = prompt_ids.to(device)[None]
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    t0 = time.monotonic()
    events[0].record()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    token = sample_next(out.logits[:, -1], generator, **sampling)
    events[1].record()
    cache = out.past_key_values
    tokens = [token]
    for _ in range(gen_tokens - 1):
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = sample_next(out.logits[:, -1], generator, **sampling)
        tokens.append(token)
    events[2].record()
    torch.cuda.synchronize(device)
    t1 = time.monotonic()

    decode_ms = events[1].elapsed_time(events[2])
    return {
        "context_tokens": ids.shape[1],
        "gen_tokens": gen_tokens,
        "ttft_ms": events[0].elapsed_time(events[1]),
        "decode_ms": decode_ms,
        "decode_tok_s": (gen_tokens - 1) / (decode_ms / 1000) if gen_tokens > 1 else float("nan"),
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "t0": t0,
        "t1": t1,
        "token_ids": torch.cat(tokens, dim=-1)[0].tolist(),
    }
