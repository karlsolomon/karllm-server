import json
import os
import time

import torch
from safetensors.torch import save_file

from config import CHUNK_SIZE, INTERACTION_DIR
from model.init import ModelState


def normalize_decoded(output):
    return "".join(output) if isinstance(output, list) else output


def generate_stream(prompt: str):
    start = time.time()
    tokens = 0
    buffer = []
    all_token_ids = []

    if not ModelState.model_ready:
        raise RuntimeError("Model not loaded")

    tokenizer = ModelState.tokenizer
    generator = ModelState.generator
    settings = ModelState.settings

    input_ids = tokenizer.encode(
        prompt, add_bos=not ModelState.session_active, add_eos=True
    )

    before_len = ModelState.cache.current_seq_len
    prompt_tokens = input_ids.shape[-1]

    if not ModelState.session_active:
        generator.begin_stream_ex(input_ids, settings)
        ModelState.session_ids = input_ids.clone()
        ModelState.session_active = True
    else:
        ModelState.session_ids = torch.cat((ModelState.session_ids, input_ids), dim=-1)
        generator._gen_feed_tokens(input_ids, settings)

    while True:
        result = generator.stream_ex()
        chunk_ids = result.get("chunk_token_ids", None)
        eos = result.get("eos", False)

        if chunk_ids is not None and chunk_ids.numel() > 0:
            new_ids = chunk_ids[0].tolist()
            all_token_ids.extend(new_ids)

            for token in new_ids:
                buffer.append(token)
                tokens += 1
                if len(buffer) >= CHUNK_SIZE:
                    decoded = tokenizer.decode(torch.tensor(buffer).unsqueeze(0))
                    yield f"data:{json.dumps({'text': decoded})}\n\n"
                    buffer.clear()

        if eos:
            break

    if buffer:
        final_decoded = tokenizer.decode(torch.tensor(buffer).unsqueeze(0))
        yield f"data:{json.dumps({'text': final_decoded})}\n\n"

    yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"

    after_len = ModelState.cache.current_seq_len
    print(
        f"[Server] Prompt tokens: {prompt_tokens}, Response tokens: {tokens}, KV delta: {after_len - before_len}"
    )

    # Save interaction snapshot
    filename = f"interaction_{int(time.time())}.safetensors"
    data = {
        "prompt_ids": input_ids.cpu(),
        "response_ids": torch.tensor(all_token_ids).cpu(),
        "start_offset": torch.tensor([before_len]),
        "end_offset": torch.tensor([after_len]),
    }
    save_file(data, os.path.join(INTERACTION_DIR, filename))
