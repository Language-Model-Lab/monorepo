import json
from pathlib import Path

import torch
from datasets import load_dataset

PROMPTS_FILE = Path(__file__).resolve().parents[2] / "prompts.json"


#WikiText-2 raw text, the dataset is joined like the standard HF perplexity recipe.
def load_wikitext2(split: str = "test") -> str:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(dataset["text"])


#Use raw text token ids without a chat template as a 1 dimensional tensor.
def tokenize_corpus(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids[0]


#The form of the fixed prompt set resembles: [{"id": ..., "text": ...}, ...].
def load_prompts() -> list[dict]:
    return json.loads(PROMPTS_FILE.read_text())


#Single turn chat prompt with the thinking mode hard switch: {"input_ids", "attention_mask"}.
def chat_inputs(tokenizer, prompt: str, enable_thinking: bool = False) -> dict:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )


#The first n non-overlapping chunks of `length` tokens, shape [n, length].
def token_chunks(ids: torch.Tensor, length: int, n: int) -> torch.Tensor:
    needed = length * n
    if ids.numel() < needed:
        raise ValueError(f"need {needed} tokens, corpus has {ids.numel()}")
    return ids[:needed].view(n, length)
