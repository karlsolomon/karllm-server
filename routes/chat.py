import json
import os
import time
from pathlib import Path

import config
import torch
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from safetensors.torch import save_file

from model.init import ModelState
from model.generation import generate_stream, snapshot_interaction, format_stream_chunk

router = APIRouter(prefix="/chat")


@router.post("/stream")
async def stream_chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return StreamingResponse(
            content=(
                f"data:{json.dumps({'text': '[Error: empty prompt]'})}\n\n" for _ in range(1)),
            media_type="text/event-stream",
        )

    ModelState.session_ids = torch.empty((1, 0), dtype=torch.long)
    ModelState.generator.begin_stream_ex(
        ModelState.session_ids, ModelState.settings)
    ModelState.session_active = True

    return StreamingResponse(generate_stream(
        prompt), media_type="text/event-stream")
