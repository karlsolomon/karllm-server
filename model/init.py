import os

import config
import exllamav2
import model
import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
    model_init,
)


class ModelState:
    """
    Container for global model state shared across API handlers.
    Includes model components, cache, and session management.
    """

    model = None
    config = None
    tokenizer = None
    generator = None
    settings = None
    cache = None
    model_ready = False
    session_active = False
    session_ids = None
    session_dir = None
    save_interactions = False


def load_model():
    """
    Initialize and load the ExLlamaV2 model, tokenizer, cache, and generator.
    """
    print("🔁 Loading model...")

    # Load model and tokenizer
    ModelState.config = exllamav2.ExLlamaV2Config(model_dir=config.MODEL_DIR)
    ModelState.config.max_seq_len = config.MODEL_MAX_SEQ_LEN
    ModelState.config.max_batch_size = 4
    ModelState.config.max_output_len = config.RESPONSE_LIMIT
    ModelState.config.max_input_len = config.PROMPT_LIMIT

    ModelState.model = exllamav2.ExLlamaV2(ModelState.config)

    # Initialize KV cache
    if config.TENSOR_PARALLEL:
        ModelState.model.load_tp(
            progress=True,
            expect_cache_tokens=config.CHAT_CONTEXT_LIMIT,
            expect_cache_base=config.CACHE_QUANTIZATION,
        )
        ModelState.cache = exllamav2.ExLlamaV2Cache_TP(
            model=ModelState.model, base=config.CACHE_QUANTIZATION
        )
    else:
        ModelState.cache = config.CACHE_QUANTIZATION(ModelState.model)

    # Configure sampling
    ModelState.settings = exllamav2.generator.ExLlamaV2Sampler().Settings()
    ModelState.settings.temperature = config.TEMPERATURE
    ModelState.settings.top_k = config.TOP_K
    ModelState.settings.top_p = config.TOP_P
    ModelState.settings.token_repetition_penalty = config.TOKEN_REPETITION_PENALTY
    ModelState.settings.length = config.RESPONSE_LIMIT

    ModelState.tokenizer = exllamav2.ExLlamaV2Tokenizer(ModelState.config)

    # Streaming generator
    ModelState.generator = exllamav2.generator.ExLlamaV2StreamingGenerator(
        model=ModelState.model,
        cache=ModelState.cache,
        tokenizer=ModelState.tokenizer,
    )
    ModelState.generator.warmup()

    ModelState.session_ids = torch.empty((1, 0), dtype=torch.long)

    print("✅ Model fully loaded.")
    ModelState.model_ready = True
