import json
import os
from pathlib import Path

import config
import exllamav2
import torch
from safetensors.torch import load_file

import model.SupportedModel as sm


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


def load_session_into_cache(session_dir: str):
    """
    Reconstruct full KV cache from interaction snapshots in a session directory.
    Each file must contain prompt_ids, response_ids, start_offset, end_offset.
    """
    session_path = Path(session_dir)
    if not session_path.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    files = sorted(
        session_path.glob("*.safetensors"),
        key=lambda f: (
            int(f.stem.split("_")[-1]) if "_" in f.stem else f.stat().st_mtime
        ),
    )

    for fpath in files:
        data = load_file(fpath)
        if "prompt_ids" not in data or "response_ids" not in data:
            continue

        prompt_ids = data["prompt_ids"]
        response_ids = data["response_ids"]

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.ndim == 1:
            response_ids = response_ids.unsqueeze(0)

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        if full_ids.ndim == 1:
            full_ids = full_ids.unsqueeze(0)

        ModelState.model.forward(
            input_ids=prompt_ids, cache=ModelState.cache, preprocess_only=True
        )

    print(
        f"✅ Loaded {
            len(files)} interactions into KV cache. Current seq_len: {
            ModelState.cache.current_seq_len}"
    )


def load_model():
    """
    Initialize and load the ExLlamaV2 model, tokenizer, cache, and generator.
    """
    print("🔁 Loading model...")

    activeModel = sm.get_active_model()
    ModelState.config = exllamav2.ExLlamaV2Config(model_dir=activeModel.path)
    ModelState.config.max_seq_len = activeModel.max_seq_len
    ModelState.config.max_batch_size = 4
    ModelState.config.max_output_len = activeModel.response_limit
    ModelState.config.max_input_len = activeModel.prompt_limit

    ModelState.model = exllamav2.ExLlamaV2(ModelState.config)

    if config.TENSOR_PARALLEL:
        ModelState.model.load_tp(
            progress=True,
            expect_cache_tokens=activeModel.max_seq_len,
            expect_cache_base=config.CACHE_QUANTIZATION,
        )
        ModelState.cache = exllamav2.ExLlamaV2Cache_TP(
            model=ModelState.model, base=config.CACHE_QUANTIZATION
        )
    else:
        ModelState.cache = config.CACHE_QUANTIZATION(ModelState.model)
        ModelState.model.load_autosplit(ModelState.cache, progress=True)

    ModelState.tokenizer = exllamav2.ExLlamaV2Tokenizer(ModelState.config)

    ModelState.settings = exllamav2.generator.ExLlamaV2Sampler.Settings()
    ModelState.settings.temperature = config.TEMPERATURE
    ModelState.settings.top_k = config.TOP_K
    ModelState.settings.top_p = config.TOP_P
    ModelState.settings.token_repetition_penalty = config.TOKEN_REPETITION_PENALTY
    ModelState.settings.length = config.RESPONSE_LIMIT
    ModelState.settings.eos_token_id = int(
        ModelState.tokenizer.eos_token_id or config.EOS_TOKEN_ID
    )
    ModelState.settings.bos_token_id = int(
        ModelState.tokenizer.bos_token_id or config.BOS_TOKEN_ID)

    ModelState.generator = exllamav2.generator.ExLlamaV2StreamingGenerator(
        model=ModelState.model,
        cache=ModelState.cache,
        tokenizer=ModelState.tokenizer,
    )

    ModelState.session_ids = torch.empty((1, 0), dtype=torch.long)
    ModelState.model_ready = True
    print("✅ Model fully loaded.")
