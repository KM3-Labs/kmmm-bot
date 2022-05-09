from typing import Optional
from pydantic import BaseModel, ValidationError, validator, root_validator


def positive(value):
    assert value > 0, "must be positive"
    return value

def positive_float_1(value: float) -> float:
    assert value >= 0.0 and value <= 1.0, "must be between 0.0 and 0.1"
    return value

class ModelLogitBiasArgs(BaseModel):
    id: int
    bias: float
    
class ModelPhraseBiasArgs(BaseModel):
    sequences: list[str]
    bias: float
    ensure_sequence_finish: bool
    generate_once: bool

class ModelGenArgs(BaseModel):
    max_length: int
    max_time: Optional[float] = None
    min_length: Optional[int] = None
    eos_token_id: Optional[int] = None
    logprobs: Optional[int] = None
    best_of: Optional[int] = None

    @validator("min_length")
    def min_length_less_than_max(cls, v, values, **kwargs):
        assert v <= values["max_length"], "must be less than max_length"
        return v
    
    @validator("logprobs")
    def logprobs_range(cls, v):
        assert v >= 0 and v <= 20, "must be between 0 and 20"
        return v
    
    _positive_items = validator("max_time", "max_length", "eos_token_id", "best_of", allow_reuse=True)(positive)


class ModelSampleArgs(BaseModel):
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    tfs: Optional[float] = None
    rep_p: Optional[float] = None
    rep_p_range: Optional[int] = None
    rep_p_slope: Optional[float] = None
    bad_words: Optional[list[str]] = None
    # logit biases are a list of int and float tuples
    logit_biases: Optional[list[ModelLogitBiasArgs]] = None
    phrase_biases: Optional[list[ModelPhraseBiasArgs]] = None

    _positive_items = validator("temp", "top_k", "rep_p_slope", "rep_p_range", allow_reuse=True)(positive)
    _positive_ranged_floats = validator("top_a", "typical_p", "tfs", allow_reuse=True)(positive_float_1)


class GPTGenRequest(BaseModel):

    prompt: str
    soft_prompt: Optional[bytes] = None
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs


    @validator("prompt")  # should soft prompt db fetching logic be done elsewhere or within the generate function?
    def softprompt_truncate(cls, v, values, **kwargs):
        if "soft_prompt" in values:
            sf = values["soft_prompt"]  # unpickle? idk where to process soft prompts but db should be done in cog
        return v

        """
        for reference:
        if softprompt:
                prompt = softprompt.get_special_str() + args["prompt"]
            else:
                prompt = args["prompt"]
        """

class Response:
    output: str
    logprobs: str


# class GPTGenProcessing(GPTGenRequest):

