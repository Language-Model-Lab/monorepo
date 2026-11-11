import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant.surgery import quantize_model_

MODEL_ID = "Qwen/Qwen3-0.6B"

# Activation dtype of the 16-bit baseline and every quantized model (task 1.5, results/dtype_check.md).
# Always load in BF16 (the checkpoint dtype), run surgery, then cast to this.
BASELINE_DTYPE = torch.float16


#revision pins the Hugging Face checkpoint (a commit hash for evidence runs); None means the latest.
def load_model(
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    model_id: str = MODEL_ID,
    revision: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
        revision=revision,
    )
    model.to(device).eval()
    return model, tokenizer


#INT8 weights quantized from the BF16 checkpoint, then cast to the activation dtype.
def load_int8_model(
    dtype: torch.dtype = BASELINE_DTYPE,
    backend: str = "reference",
    device: str = "cuda",
    model_id: str = MODEL_ID,
    revision: str | None = None,
):
    model, tokenizer = load_model(torch.bfloat16, device=device, model_id=model_id, revision=revision)
    quantize_model_(model, backend=backend)
    model.to(dtype)
    return model, tokenizer

