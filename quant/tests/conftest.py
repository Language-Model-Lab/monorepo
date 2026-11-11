import os

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

# Validation runs set QUANT_REQUIRE_CUDA=1 so that a machine without a usable GPU cannot "pass" by skipping.
REQUIRE_CUDA = os.environ.get("QUANT_REQUIRE_CUDA") == "1"


def pytest_sessionstart(session):
    if REQUIRE_CUDA and not torch.cuda.is_available():
        pytest.exit("QUANT_REQUIRE_CUDA=1 but torch.cuda.is_available() is False", returncode=1)


def _allowed_skip(report) -> bool:
    # Hardware dependent: multi-GPU tests may skip on a single GPU machine. Every other skip is unexpected.
    return "multi_gpu" in getattr(report, "keywords", {}) and torch.cuda.device_count() < 2


def pytest_sessionfinish(session, exitstatus):
    if not REQUIRE_CUDA:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    unexpected = [report.nodeid for report in skipped if not _allowed_skip(report)]
    if unexpected:
        if reporter is not None:
            reporter.write_line(f"QUANT_REQUIRE_CUDA=1: {len(unexpected)} unexpected skip(s): {unexpected}", red=True)
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


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

