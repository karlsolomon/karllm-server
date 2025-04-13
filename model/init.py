import os

# from exllamav2.cache import ExLlamaV2Cache_TP
# from exllamav2.embedding import ExLlamaV2Embedding
# from exllamav2.generator import ExLlamaV2Sampler
# from exllamav2.generator.streaming import ExLlamaV2StreamingGenerator
# from exllamav2.model_init import init as model_init
#
import config
import exllamav2


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
    session_ids = None
    session_active = False


def load_model():
    """
    Initialize and load the ExLlamaV2 model, tokenizer, cache, and generator.
    """
    print("🔁 Loading model...")

    # Load model and tokenizer
    ModelState.config = exllamav2.ExLlamaV2Config(model_dir=config.MODEL_DIR)

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
        ModelState.cache = exllamav2.config.CACHE_QUANTIZATION(ModelState.model)

    # Configure sampling
    ModelState.settings = exllamav2.generator.ExLlamaV2Sampler().Settings()
    ModelState.settings.temperature = config.TEMPERATURE
    ModelState.settings.top_k = config.TOP_K
    ModelState.settings.top_p = config.TOP_P
    ModelState.settings.token_repetition_penalty = config.TOKEN_REPETITION_PENALTY
    ModelState.settings.length = config.RESPONSE_LIMIT

    ModelState.tokenizer = exllamav2.ExLlamaV2Tokenizer(ModelState.config)

    # Build streaming generator
    ModelState.generator = exllamav2.generator.ExLlamaV2DynamicGenerator(
        model=ModelState.model, cache=ModelState.cache, tokenizer=ModelState.tokenizer
    )

    ModelState.generator.warmup()

    print("✅ Model fully loaded.")
    ModelState.model_ready = True
