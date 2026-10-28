import time
startup_start = time.perf_counter()

from fastapi import FastAPI
from kernel_backend.schemas import GenerateRequest, GenerateResponse
from kernel_backend.model import load_model
from kernel_backend.generation import generate_text, run_generation

app = FastAPI()

#Load model only once when backend is started
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
print(f"Startup through model loading: {time.perf_counter() - startup_start:.2f}s")

@app.post("/generate", response_model=GenerateResponse)
def generate(request:GenerateRequest):

    prompt = request.prompt
    configuration_a = request.configuration_a
    configuration_b = request.configuration_b
    result_a = run_generation(model, tokenizer, prompt, configuration_a)
    result_b = run_generation(model, tokenizer, prompt, configuration_b)

    return GenerateResponse(
        condition_a_text = result_a.generated_text,
        condition_b_text = result_b.generated_text,
        condition_a_latency_ms = result_a.latency_ms,
        condition_b_latency_ms = result_b.latency_ms,
        condition_a_tokens_per_second = result_a.tokens_per_second,
        condition_b_tokens_per_second = result_b.tokens_per_second,
    )