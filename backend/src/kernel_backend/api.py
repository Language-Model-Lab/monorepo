from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from kernel_backend.errors import InvalidLayerError, InvalidHeadError
from kernel_backend.schemas import GenerateRequest, GenerateResponse
from kernel_backend.model import load_model
from kernel_backend.generation import generate_text, run_generation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load model only once when backend is started
model, tokenizer = load_model("Qwen/Qwen3-0.6B")

@app.post("/generate", response_model=GenerateResponse)
def generate(request:GenerateRequest):

    prompt = request.prompt
    configuration_a = request.configuration_a
    configuration_b = request.configuration_b

    #Generate text using first configuration
    try:
        result_a = run_generation(model, tokenizer, prompt, configuration_a)

    except (InvalidLayerError, InvalidHeadError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Configuration A: {exc}",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Configuration A: model generation failed.",
        ) from exc
    
    #Generate text using second configuration
    try:
        result_b = run_generation(model, tokenizer, prompt, configuration_b)

    except (InvalidLayerError, InvalidHeadError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Configuration B: {exc}",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Configuration B: model generation failed.",
        ) from exc

    return GenerateResponse(
        condition_a_text = result_a.generated_text,
        condition_b_text = result_b.generated_text,
        condition_a_latency_ms = result_a.latency_ms,
        condition_b_latency_ms = result_b.latency_ms,
        condition_a_tokens_per_second = result_a.tokens_per_second,
        condition_b_tokens_per_second = result_b.tokens_per_second,
    )