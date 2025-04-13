import json
import os
import time

import exllamav2
import torch
from exllamav2 import ExLlamaV2Cache_Q8, ExLlamaV2Cache_TP
from safetensors.torch import save_file

import config
from model.init import ModelState


def normalize_decoded(output):
    """
    Normalize decoded output from model generator.
    Handles both string and list-of-strings formats.
    """
    return "".join(output) if isinstance(output, list) else output


def generate_stream(prompt: str):
    """
    Stream token-by-token output from the model given an input prompt.
    Yields JSON-encoded SSE (Server-Sent Events) chunks suitable for FastAPI StreamingResponse.

    Also captures prompt/response token IDs and saves session metadata to .safetensors.
    """
    start = time.time()
    text = None

    if not ModelState.model_ready:
        raise RuntimeError("Model not loaded")

    # Send Job
    input_ids = ModelState.tokenizer.encode(prompt, add_bos=True, add_eos=True)
    job = exllamav2.generator.ExLlamaV2DynamicJob(
        input_ids=input_ids,
        max_new_tokens=config.RESPONSE_LIMIT,
        gen_settings=ModelState.settings,
        filter_prefer_eos=True,
    )
    job.stop_tokens.add(config.EOS_TOKEN_ID)
    ModelState.generator.enqueue(job)
    eos = False
    while not eos:
        results = ModelState.generator.iterate()
        for result in results:
            assert result["job"] == job
            if result["stage"] == "streaming":
                text = result.get("text", "")
                yield f"data:{json.dumps({'text': text})}\n\n"
                if result["eos"]:
                    eos = True
                    yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"
