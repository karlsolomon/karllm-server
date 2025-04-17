import json
import os
import time
from pathlib import Path

import exllamav2
import torch
from exllamav2 import ExLlamaV2Cache_Q8, ExLlamaV2Cache_TP
from safetensors.torch import save_file

import config
from model.init import ModelState

buffer = []
ModelState.session_id = 0


def start_stream():
    # ModelState.cache.reset() # TODO: investigate why this causes model to talk to itself.
    # ModelState.session_ids = torch.empty((1, 0), dtype=torch.long)
    ModelState.generator.begin_stream_ex(ModelState.session_ids, ModelState.settings)
    ModelState.session_active = True


def encode_file(path: Path):
    text = path.read_text(encoding="utf-8")

    all_ids = ModelState.tokenizer.encode(text, add_bos=False, add_eos=False)
    if all_ids.ndim == 1:
        all_ids = all_ids.unsqueeze(0)

    chunk_size = 16 * 1024
    injected_tokens = 0

    for i in range(0, all_ids.shape[1], chunk_size):
        chunk = all_ids[:, i : i + chunk_size]
        ModelState.model.forward(
            input_ids=chunk, cache=ModelState.cache, preprocess_only=True
        )
        injected_tokens += chunk.shape[1]

    return injected_tokens


def normalize_decoded(output):
    """
    Normalize decoded output from model generator.
    Handles both string and list-of-strings formats.
    """
    return "".join(output) if isinstance(output, list) else output


def continue_prompt(prompt: str):
    # Send Job
    all_token_ids = []
    before_len = ModelState.cache.current_seq_len
    input_ids = ModelState.tokenizer.encode(
        prompt, add_bos=not ModelState.session_active, add_eos=True
    )
    token_count = 0
    start_time = time.perf_counter()
    ModelState.generator._gen_feed_tokens(input_ids, ModelState.settings)
    while True:
        result = ModelState.generator.stream_ex()
        chunk_ids = result.get("chunk_token_ids", None)
        token_count += chunk_ids.numel()

        eos = result.get("eos", False)
        if chunk_ids is not None and chunk_ids.numel() > 0:
            new_ids = chunk_ids[0].tolist()
            all_token_ids.extend(new_ids)

            # ⛔ Stop if EOS is in output
            if (
                (ModelState.tokenizer.eos_token_id in new_ids)
                or (config.EOS_TOKEN_ID in new_ids)
                or (config.EOS_TOKEN_ID_BACKUP in new_ids)
            ):
                print("✅ Detected EOS token in output — stopping early.")
                eos = True

            for token in new_ids:
                buffer.append(token)
                if len(buffer) >= config.CHUNK_SIZE:
                    decoded = ModelState.tokenizer.decode(
                        torch.tensor(buffer).unsqueeze(0)
                    )
                    yield f"data:{json.dumps({'text': decoded})}\n\n"
                    buffer.clear()
        if eos:
            break
    if buffer:
        final_decoded = ModelState.tokenizer.decode(torch.tensor(buffer).unsqueeze(0))
        buffer.clear()
        yield f"data:{json.dumps({'text': final_decoded})}\n\n"
    duration = time.perf_counter() - start_time
    yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"
    print(f"⏱️ {token_count} tokens @ {token_count/duration:.2f} tokens/s.")

    if ModelState.save_interactions:
        # Save interaction snapshot
        f = f"{int(time.time())}.safetensors"
        after_len = ModelState.cache.current_seq_len
        data = {
            "prompt_ids": input_ids.cpu(),
            "response_ids": torch.tensor(all_token_ids).cpu(),
            "start_offset": torch.tensor([before_len]),
            "end_offset": torch.tensor([after_len]),
        }
        save_file(data, os.path.join(ModelState.session_dir, f))
