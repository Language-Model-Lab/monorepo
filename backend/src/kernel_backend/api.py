from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from kernel_backend.errors import InvalidLayerError, InvalidHeadError
from kernel_backend.schemas import GenerateRequest, GenerateResponse, GenerationError, GenerationSuccess
from kernel_backend.model import load_model
from kernel_backend.generation import generate_text, run_generation
import threading

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

#Concurrency management lock
generation_lock = threading.Lock()

#Load model only once when backend is started
model, tokenizer = load_model("Qwen/Qwen3-0.6B")

@app.post("/generate", response_model=GenerateResponse)
def generate(request:GenerateRequest):

    prompt = request.prompt
    configuration_a = request.configuration_a
    configuration_b = request.configuration_b

    #### START LOCK HERE #####
    with generation_lock:
        #Generate text using first configuration
        try:
            result_a = run_generation(model, tokenizer, prompt, configuration_a)

            response_a = GenerationSuccess(
                state = "success",
                text = result_a.generated_text,
                latency_ms= result_a.latency_ms,
                num_generated_tokens = result_a.num_generated_tokens,
                tokens_per_second = result_a.tokens_per_second
                )

        except InvalidLayerError as exc:
            response_a = GenerationError(
                state = "error",
                error_type = "invalid_layer",
                message = str(exc)
                )
        
        except InvalidHeadError as exc:
            response_a = GenerationError(
                state = "error",
                error_type = "invalid_head",
                message = str(exc)
                )

        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Configuration A: model generation failed.",
            ) from exc
        
        #Generate text using second configuration
        try:
            result_b = run_generation(model, tokenizer, prompt, configuration_b)

            response_b = GenerationSuccess(
                state = "success",
                text = result_b.generated_text,
                latency_ms= result_b.latency_ms,
                num_generated_tokens = result_b.num_generated_tokens,
                tokens_per_second = result_b.tokens_per_second
                )

        except InvalidLayerError as exc:
            response_b = GenerationError(
                state = "error",
                error_type = "invalid_layer",
                message = str(exc)
                )
        
        except InvalidHeadError as exc:
            response_b = GenerationError(
                state = "error",
                error_type = "invalid_head",
                message = str(exc)
                )

        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Configuration B: model generation failed.",
            ) from exc
    #### RELEASE LOCK HERE #####

    return GenerateResponse(
        condition_a = response_a,
        condition_b = response_b,
    )