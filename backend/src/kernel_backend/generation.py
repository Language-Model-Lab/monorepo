from kernel_backend.interventions import add_head_ablation, add_mlp_ablation
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    StoppingCriteria, 
    StoppingCriteriaList
    )
from kernel_backend.schemas import (
   GenerationSettings,
   InterventionSettings,
   GenerateRequest,
   GenerateResponse,
   ExperimentConfiguration
   )

class SentenceBoundaryCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length, soft_limit):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.soft_limit = soft_limit
    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[:, self.prompt_length:]

        if generated_ids.shape[1] < self.soft_limit:
            return torch.zeros(
                input_ids.shape[0],
                dtype=torch.bool,
                device=input_ids.device,
                )
        
        stop_conditions = []
        for sequence in generated_ids:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            stop_conditions.append(text.rstrip().endswith(('.', '?', '!')))
        
        return torch.tensor(
                stop_conditions,
                dtype=torch.bool,
                device=input_ids.device,
                )


def run_generation(model, tokenizer, prompt: str, request:ExperimentConfiguration):
    generation_settings = request.generation
    intervention_settings = request.intervention
    handles = []

    try:
        if intervention_settings is not None:
            if intervention_settings.type == "mlp":
                for layer_idx in intervention_settings.layers:
                    handle, _, _, _ = add_mlp_ablation(
                        model=model, 
                        layer_idx=layer_idx, 
                        proportion= intervention_settings.proportion, 
                        seed=intervention_settings.seed
                        )
                    handles.append(handle)
            elif intervention_settings.type == "attention":
                for layer_idx in intervention_settings.layers:
                    handle, _, _ = add_head_ablation(
                        model=model, 
                        layer_idx=layer_idx, 
                        head_indices= intervention_settings.head_idxs, 
                        )
                    handles.append(handle)
                        
        model_response = generate_text(
                                model=model, 
                                tokenizer=tokenizer,
                                prompt=prompt, 
                                settings=generation_settings
                                )
    finally:
        for handle in handles:
            handle.remove()
            
    return model_response
    

    

def generate_text(model, tokenizer, prompt:str, settings: GenerationSettings) -> str:
    messages = [
    {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=settings.enable_thinking,
    ).to(model.device)

    #Absolute maximum amount of new tokens the model 
    #is allowed to generate.
    hard_limit = int(settings.max_new_tokens*1.25)

    #Custom stopping criteria to encourage stopping generation at 
    #more natural points in sentences, rather than mid-sentence.
    criteria = SentenceBoundaryCriteria(
            tokenizer=tokenizer, 
            prompt_length=inputs["input_ids"].shape[-1],
            soft_limit=settings.max_new_tokens
            )

    generation_kwargs = {
        "max_new_tokens": hard_limit,
        "do_sample": settings.do_sample,
        "repetition_penalty": settings.repetition_penalty,
        "stopping_criteria": [criteria]
        }

    if settings.do_sample:
        generation_kwargs["temperature"] = settings.temperature
        generation_kwargs["top_k"] = settings.top_k
        generation_kwargs["top_p"] = settings.top_p
    
    output = model.generate(
        **inputs,
        **generation_kwargs,
        )

    return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
            )


     