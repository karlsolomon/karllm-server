import os

from exllamav2 import ExLlamaV2Cache_Q8
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator.streaming import ExLlamaV2StreamingGenerator
from exllamav2.model_init import init as model_init

from config import MODEL_DIR, PROMPT_LIMIT, RESPONSE_LIMIT

model = None
tokenizer = None
generator = None
settings = None
cache = None


class ModelState:
    model = None
    tokenizer = None
    generator = None
    settings = None
    cache = None
    model_ready = False
    # persistence
    session_ids = None
    session_active = False


def lazy_load_model():
    print("🔁 Loading model...")
    args = type(
        "Args",
        (),
        {
            "model_dir": MODEL_DIR,
            "gpu_split": None,
            "tensor_parallel": False,
            "length": 4096,
            "rope_scale": None,
            "rope_alpha": None,
            "rope_yarn": None,
            "no_flash_attn": False,
            "no_xformers": False,
            "no_sdpa": False,
            "no_graphs": False,
            "low_mem": False,
            "experts_per_token": None,
            "load_q4": False,
            "load_q8": True,
            "fast_safetensors": False,
            "ignore_compatibility": True,
            "chunk_size": PROMPT_LIMIT,
        },
    )()

    ModelState.model, ModelState.tokenizer = model_init(
        args, progress=True, max_input_len=PROMPT_LIMIT, max_output_len=RESPONSE_LIMIT
    )

    ModelState.settings = ExLlamaV2Sampler().Settings()
    ModelState.settings.temperature = 0.8
    ModelState.settings.top_k = 50
    ModelState.settings.top_p = 0.95
    ModelState.settings.token_repetition_penalty = 1.1
    ModelState.settings.eos_token_id = ModelState.tokenizer.eos_token_id or 151643

    ModelState.cache = ExLlamaV2Cache_Q8(
        ModelState.model, lazy=not ModelState.model.loaded
    )
    ModelState.generator = ExLlamaV2StreamingGenerator(
        ModelState.model, ModelState.cache, ModelState.tokenizer
    )
    print("✅ Model fully loaded.")
    ModelState.model_ready = True
