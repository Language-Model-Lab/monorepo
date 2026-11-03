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
from kernel_backend.errors import InvalidLayerError, InvalidHeadError
from kernel_backend.generation import GenerationResult
import torch
from kernel_backend import generation
from kernel_backend import model as model_module
from unittest.mock import MagicMock
import pytest
import time
import threading
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor

def test_api_generation(monkeypatch):
    def fake_load_model(model_name: str, device_map: str = "auto"):
        model = MagicMock()
        tokenizer = MagicMock()
        return model, tokenizer

    monkeypatch.setattr(
        model_module,
        "load_model",
        fake_load_model,
    )

    generation_settings = GenerationSettings()
    request = ExperimentConfiguration(generation=generation_settings)
    
    def fake_run_generation(model, tokenizer, prompt, request):
        return GenerationResult(
            generated_text="fake response",
            latency_ms=100.0,
            num_generated_tokens=20,
            tokens_per_second=200.0,
        )
    
    from kernel_backend import api

    monkeypatch.setattr(api, "run_generation", fake_run_generation)
    client = TestClient(api.app)

    response = client.post(
        "/generate",
        json={
            "prompt": "test",
            "configuration_a": request.model_dump(),
            "configuration_b": request.model_dump(),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["condition_a"]["text"] == 'fake response'
    assert data["condition_b"]["text"] == 'fake response'

def test_concurrent_api_generation(monkeypatch):
    def fake_load_model(model_name: str, device_map: str = "auto"):
        model = MagicMock()
        tokenizer = MagicMock()
        return model, tokenizer

    monkeypatch.setattr(
        model_module,
        "load_model",
        fake_load_model,
    )

    active_processes = 0
    max_active_processes = 0
    execution_order = []
    counter_lock = threading.Lock()

    def record_start():
        nonlocal active_processes, max_active_processes
        with counter_lock:
            active_processes += 1
            max_active_processes = max(max_active_processes, active_processes)

    def record_end():
        nonlocal active_processes
        with counter_lock:
            active_processes -= 1
    
    def fake_run_generation(model, tokenizer, prompt, request):
        with counter_lock:
            execution_order.append(prompt)
        record_start()
        time.sleep(0.1)
        record_end()
        return GenerationResult(
            generated_text="fake response",
            latency_ms=100.0,
            num_generated_tokens=20,
            tokens_per_second=200.0,
        )
    
    from kernel_backend import api
    monkeypatch.setattr(api, "run_generation", fake_run_generation)
    client = TestClient(api.app)

    generation_settings = GenerationSettings()
    request = ExperimentConfiguration(generation=generation_settings)

    def send_request(prompt):
        return client.post(
            "/generate",
            json={
                "prompt": prompt,
                "configuration_a": request.model_dump(),
                "configuration_b": request.model_dump(),
            },
        )    

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(send_request, "1")
        future_2 = executor.submit(send_request, "2")

        response_1 = future_1.result()
        response_2 = future_2.result()
        
    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert max_active_processes == 1
    assert execution_order in [
        ["1", "1", "2", "2"],
        ["2", "2", "1", "1"],
    ]

def test_mixed_success_api_generation(monkeypatch):
    def fake_load_model(model_name: str, device_map: str = "auto"):
        model = MagicMock()
        tokenizer = MagicMock()
        return model, tokenizer

    monkeypatch.setattr(
        model_module,
        "load_model",
        fake_load_model,
    )

    generation_settings = GenerationSettings()
    request = ExperimentConfiguration(
            generation=generation_settings,
        )
    
    num_call = 0
    def fake_run_generation(model, tokenizer, prompt, request):
        nonlocal num_call
        num_call += 1
        if num_call == 1:
            raise InvalidHeadError("Invalid head")
        else:
            return GenerationResult(
                generated_text="fake response",
                latency_ms=100.0,
                num_generated_tokens=20,
                tokens_per_second=200.0,
            )
    
    from kernel_backend import api

    monkeypatch.setattr(api, "run_generation", fake_run_generation)
    client = TestClient(api.app)

    response = client.post(
        "/generate",
        json={
            "prompt": "test",
            "configuration_a": request.model_dump(),
            "configuration_b": request.model_dump(),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["condition_a"]["state"] == 'error'
    assert data["condition_b"]["state"] == 'success'