from kernel_backend.schemas import (
GenerationSettings,
InterventionSettings,
GenerateRequest,
GenerateResponse,
ExperimentConfiguration,
MLPInterventionSettings,
AttentionInterventionSettings
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    StoppingCriteria, 
    StoppingCriteriaList
    )
import torch
from kernel_backend import generation
from unittest.mock import MagicMock
import pytest

class FakeTokenizer:
    def __init__(self, decoded_text):
        self.decoded_text = decoded_text

    def decode(self, sequence, skip_special_tokens=True):
        return self.decoded_text

def test_SentenceBoundaryCriteria_good_end_char():
    fake_tokenizer = FakeTokenizer("This is a complete sentence.")
    criteria = generation.SentenceBoundaryCriteria(
        tokenizer=fake_tokenizer,
        prompt_length=3,
        soft_limit=5,
        )

    input_ids = torch.tensor([[10, 20, 30, 101, 102, 103, 104, 105]])
    result = criteria(input_ids, scores=None)
    assert result == True

def test_SentenceBoundaryCriteria_short_input():
    fake_tokenizer = FakeTokenizer("This is a complete sentence.")
    criteria = generation.SentenceBoundaryCriteria(
        tokenizer=fake_tokenizer,
        prompt_length=3,
        soft_limit=5,
        )

    input_ids = torch.tensor([[10, 20, 30, 101, 102, 103, 104]])
    result = criteria(input_ids, scores=None)
    assert result == False

def test_SentenceBoundaryCriteria_no_end_char():
    fake_tokenizer = FakeTokenizer("This is an incomplete sentence")
    criteria = generation.SentenceBoundaryCriteria(
        tokenizer=fake_tokenizer,
        prompt_length=3,
        soft_limit=5,
        )
    input_ids = torch.tensor([[10, 20, 30, 101, 102, 103, 104, 105]])
    result = criteria(input_ids, scores=None)
    assert result == False

    input_ids = torch.tensor([[10, 20, 30, 101, 102, 103, 104]])
    result = criteria(input_ids, scores=None)
    assert result == False

def test_run_generation_no_intervention(monkeypatch):
    model = MagicMock()
    tokenizer = MagicMock()
    prompt = "test"
    generation_settings = GenerationSettings()
    request = ExperimentConfiguration(generation=generation_settings)

    handle = MagicMock()

    def fake_generate_text(model, tokenizer, prompt, settings):
        return "fake response"

    monkeypatch.setattr(
        generation,
        "generate_text",
        fake_generate_text,
    )

    model_response = generation.run_generation(
                        model=model, 
                        tokenizer=tokenizer,
                        prompt=prompt,
                        request=request
                        )

    assert model_response == "fake response"

def test_run_generation_mlp_intervention(monkeypatch):
    model = MagicMock()
    tokenizer = MagicMock()
    prompt = "test"
    generation_settings = GenerationSettings()
    intervention_settings = MLPInterventionSettings(
                                                    type="mlp", 
                                                    layers=[0,1,2,3],
                                                    proportion=0.5,
                                                    seed=42)
    request = ExperimentConfiguration(
                                        generation=generation_settings, 
                                        intervention=intervention_settings
                                        )

    handle = MagicMock()

    def fake_add_mlp_ablation(model, layer_idx, proportion, seed):
        return handle, None, layer_idx, proportion

    def fake_generate_text(model, tokenizer, prompt, settings):
        return "fake response"

    monkeypatch.setattr(
        generation,
        "generate_text",
        fake_generate_text,
    )

    monkeypatch.setattr(
        generation,
        "add_mlp_ablation",
        fake_add_mlp_ablation,
    )

    model_response = generation.run_generation(
                        model=model, 
                        tokenizer=tokenizer,
                        prompt=prompt,
                        request=request
                        )

    assert handle.remove.call_count == 4

def test_run_generation_attention_intervention(monkeypatch):
    model = MagicMock()
    tokenizer = MagicMock()
    prompt = "test"
    generation_settings = GenerationSettings()
    intervention_settings = AttentionInterventionSettings(
                                                    type="attention", 
                                                    layers=[0,1,2,3],
                                                    head_idxs=[0,1,2])
    request = ExperimentConfiguration(
                                        generation=generation_settings, 
                                        intervention=intervention_settings
                                        )

    handle = MagicMock()

    def fake_add_head_ablation(model, head_indices, layer_idx):
        return handle, head_indices, layer_idx

    def fake_generate_text(model, tokenizer, prompt, settings):
        return "fake response"

    monkeypatch.setattr(
        generation,
        "generate_text",
        fake_generate_text,
    )

    monkeypatch.setattr(
        generation,
        "add_head_ablation",
        fake_add_head_ablation,
    )

    model_response = generation.run_generation(
                        model=model, 
                        tokenizer=tokenizer,
                        prompt=prompt,
                        request=request
                        )

    assert handle.remove.call_count == 4

def test_run_generation_finally_behavior(monkeypatch):
    model = MagicMock()
    tokenizer = MagicMock()
    prompt = "test"
    generation_settings = GenerationSettings()
    intervention_settings = AttentionInterventionSettings(
                                                    type="attention", 
                                                    layers=[0,1,2,3],
                                                    head_idxs=[0,1,2])
    request = ExperimentConfiguration(
                                        generation=generation_settings, 
                                        intervention=intervention_settings
                                        )

    handle = MagicMock()

    def fake_add_head_ablation(model, head_indices, layer_idx):
        return handle, head_indices, layer_idx

    def fake_generate_text(model, tokenizer, prompt, settings):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        generation,
        "generate_text",
        fake_generate_text,
    )

    monkeypatch.setattr(
        generation,
        "add_head_ablation",
        fake_add_head_ablation,
    )

    with pytest.raises(RuntimeError):    
        generation.run_generation(
                                model=model, 
                                tokenizer=tokenizer,
                                prompt=prompt,
                                request=request
                                )
    assert handle.remove.call_count == 4