import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_model():
    """A randomly initialized Qwen3 with the real model's module names and quirks
    (head_dim != hidden_size / heads, GQA, tied embeddings), small enough for CPU."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
