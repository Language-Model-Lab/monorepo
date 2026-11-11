from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class GenerationSuccess(StrictModel):
    state: Literal["success"]
    text: str
    latency_ms: float
    num_generated_tokens: int
    tokens_per_second: float
    
class GenerationError(StrictModel):
    state: Literal["error"]
    error_type: str
    message: str

class GenerateResponse(StrictModel):   
    condition_a: GenerationSuccess | GenerationError
    condition_b: GenerationSuccess | GenerationError

class GenerateRequest(StrictModel):
    prompt: str = Field(min_length = 1, max_length=2000)
    configuration_a: ExperimentConfiguration
    configuration_b: ExperimentConfiguration

class ExperimentConfiguration(StrictModel):
    generation: GenerationSettings
    system_prompt: str | None = Field(default=None, max_length=500)
    intervention: InterventionSettings | None = None
    
class GenerationSettings(StrictModel):
    max_new_tokens: int = Field(default=50, ge=1, le=100)
    do_sample: bool = False
    temperature: float = Field(default=0.8, gt=0.0, le=10) 
    top_k: int = Field(default=50, ge=0, le=100)
    top_p: float = Field(default=0.95, ge=0.0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0.0, le=50)
    enable_thinking: bool = False  

class MLPInterventionSettings(StrictModel):
    type: Literal["mlp"]
    layers: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    proportion: float = Field(default=0.25, ge=0.0, le=1.0)
    seed: int = 42
    scale: float = 0.0

    @field_validator("layers")
    @classmethod
    def layers_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("layers must contain unique values.")
        return value

class AttentionInterventionSettings(StrictModel):
    type: Literal["attention"]
    layers: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    head_idxs: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    scale: float = 0.0

    @field_validator("head_idxs")
    @classmethod
    def head_idxs_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("head_idxs must contain unique values.")
        return value
    
    @field_validator("layers")
    @classmethod
    def layers_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("layers must contain unique values.")
        return value

InterventionSettings = Annotated[MLPInterventionSettings|AttentionInterventionSettings,
                        Field(discriminator='type')]

