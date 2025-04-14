import json
import os
import time

import config
import exllamav2
import torch
from exllamav2 import ExLlamaV2Cache_Q8, ExLlamaV2Cache_TP
from model.init import ModelState
from safetensors.torch import save_file


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
    if not ModelState.model_ready:
        raise RuntimeError("Model not loaded")

    # Send Job
    input_ids = ModelState.tokenizer.encode(
        prompt, add_bos=not ModelState.session_active, add_eos=True
    )
    buffer = []
    ModelState.session_ids = torch.cat([ModelState.session_ids, input_ids], dim=-1)

    job = exllamav2.generator.ExLlamaV2DynamicJob(
        input_ids=ModelState.session_ids,
        max_new_tokens=config.RESPONSE_LIMIT,
        gen_settings=ModelState.settings,
        filter_prefer_eos=True,
    )
    job.stop_tokens.add(config.EOS_TOKEN_ID)
    ModelState.generator.enqueue(job)

    ModelState.session_active = True

    while ModelState.generator.num_remaining_jobs():
        results = ModelState.generator.iterate()
        if len(results) >= 1:
            r = results[len(results) - 1]
            if r["stage"] == "streaming":
                buffer.append(r.get("text", ""))
                if r["eos"] or (len(buffer) >= config.CHUNK_SIZE):
                    yield f"data:{json.dumps({'text': "".join(buffer)})}\n\n"
                    buffer.clear()
            if r["eos"]:
                yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"
                ttft = r["time_prefill"]
                output_tokens = r["new_tokens"]
                tps = output_tokens / (r["time_generate"] - ttft)
                print(f"🕐 TPS: {tps:.3f} in {input_ids.shape[1]} out {output_tokens}")
