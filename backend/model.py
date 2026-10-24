from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def load_model(model_name: str, device_map: str="auto"):
    total_start = time.perf_counter()
    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer: {time.perf_counter() - start:.2f}s")
    
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    print(f"Model from_pretrained: {time.perf_counter() - start:.2f}s")
    print(f"Total load_model: {time.perf_counter() - total_start:.2f}s")
    
    return model, tokenizer