import time
startup_start = time.perf_counter()

from fastapi import FastAPI
from schemas import GenerateRequest, GenerateResponse
from model import load_model
from generation import generate_text, run_generation

app = FastAPI()

#Load model only once when backend is started
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
print(f"Startup through model loading: {time.perf_counter() - startup_start:.2f}s")

@app.post("/generate", response_model=GenerateResponse)
def generate(request:GenerateRequest):

    prompt = request.prompt
    configuration_a = request.configuration_a
    configuration_b = request.configuration_b
    generated_text_a = run_generation(model, tokenizer, prompt, configuration_a)
    generated_text_b = run_generation(model, tokenizer, prompt, configuration_b)

    return GenerateResponse(
        condition_a_text= generated_text_a,
        condition_b_text= generated_text_b,
        condition_a_latency_ms=3.33,
        condition_b_latency_ms=123.123,
    )