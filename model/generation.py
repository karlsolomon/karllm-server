import json
import os
import time
from pathlib import Path

import config
import torch
from safetensors.torch import save_file

from model.init import ModelState


def start_stream():
    # TODO: investigate why this causes model to talk to itself.
    ModelState.cache.reset()
    ModelState.session_ids = torch.empty((1, 0), dtype=torch.long)
    ModelState.generator.stop_tokens.add(config.EOS_TOKEN_ID)
    ModelState.generator.stop_tokens.add(config.BOS_TOKEN_ID)
    ModelState.generator.stop_tokens.add(config.EOS_TOKEN_ID_BACKUP)
    print(
        f"stop tokens = {
            ModelState.generator.stop_tokens}: {
            ModelState.tokenizer.decode(
                torch.Tensor(list(ModelState.generator.stop_tokens)))}")
    ModelState.generator.begin_stream_ex(
        ModelState.session_ids, ModelState.settings)
    ModelState.session_active = True


def encode_file(path: Path):
    text = path.read_text(encoding="utf-8")

    all_ids = ModelState.tokenizer.encode(text, add_bos=False, add_eos=False)
    if all_ids.ndim == 1:
        all_ids = all_ids.unsqueeze(0)

    chunk_size = 16 * 1024
    injected_tokens = 0

    for i in range(0, all_ids.shape[1], chunk_size):
        chunk = all_ids[:, i:i + chunk_size]
        ModelState.model.forward(
            input_ids=chunk, cache=ModelState.cache, preprocess_only=True
        )
        injected_tokens += chunk.shape[1]

    return injected_tokens


def format_stream_chunk(token_ids):
    decoded = ModelState.tokenizer.decode(torch.tensor(token_ids).unsqueeze(0))
    return f"data:{json.dumps({'text': decoded})}\n\n"


def snapshot_interaction(input_ids, response_ids, before_len):
    f = f"{int(time.time())}.safetensors"
    after_len = ModelState.cache.current_seq_len
    data = {
        "prompt_ids": input_ids.cpu(),
        "response_ids": torch.tensor(response_ids).cpu(),
        "start_offset": torch.tensor([before_len]),
        "end_offset": torch.tensor([after_len]),
    }
    save_file(data, os.path.join(ModelState.session_dir, f))


def generate_stream(prompt: str):
    buffer = []
    all_token_ids = []

    input_ids = ModelState.tokenizer.encode(
        prompt, add_bos=not ModelState.session_active, add_eos=True
    )
    before_len = ModelState.cache.current_seq_len
    ModelState.generator._gen_feed_tokens(input_ids, ModelState.settings)

    token_count = 0
    start_time = time.perf_counter()

    while True:
        result = ModelState.generator.stream_ex()
        chunk_ids = result.get("chunk_token_ids", None)
        token_count += chunk_ids.numel()
        eos = result.get("eos", False)

        if chunk_ids is not None and chunk_ids.numel() > 0:
            new_ids = chunk_ids[0].tolist()
            print(new_ids)
            all_token_ids.extend(new_ids)
            buffer.extend(new_ids)
            for token in ModelState.generator.stop_tokens:
                if (token in new_ids):
                    print("✅ Detected EOS token in output — stopping early.")
                    eos = True
            if len(buffer) >= config.CHUNK_SIZE:
                yield format_stream_chunk(buffer)
                buffer.clear()
        if eos:
            break

    if buffer:
        yield format_stream_chunk(buffer)

    duration = time.perf_counter() - start_time
    yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"
    print(f"⏱️ {token_count} tokens @ {token_count / duration:.2f} tokens/s.")

    if ModelState.save_interactions:
        snapshot_interaction(input_ids, all_token_ids, before_len)
