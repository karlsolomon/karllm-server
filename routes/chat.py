import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import SAFETENSORS_FILE, SESSION_CACHE_FILE
from context.memory import chat_context, instruction_context, trim_instruction_context
from model.checkpoint import load_checkpoint, save_checkpoint
from model.generation import generate_stream
from model.init import ModelState
from schema import ChatRequest

chat_router = APIRouter()


@chat_router.post("/send")
async def send_chat(request: ChatRequest):
    chat_context.append(request.prompt.strip())
    return JSONResponse(content={"message": "Prompt received."})


@chat_router.post("/session/dump")
async def dump_session():
    if not ModelState.session_active:
        return JSONResponse(content={"message": "No active session to dump."})

    session_data = {
        "session_ids": ModelState.session_ids,
        "current_seq_len": ModelState.cache.current_seq_len,
        "kv_cache": {
            "key_states": [k.clone().cpu() for k in ModelState.cache.key_states],
            "value_states": [v.clone().cpu() for v in ModelState.cache.value_states],
        },
    }
    torch.save(session_data, SESSION_CACHE_FILE)
    return JSONResponse(content={"message": f"Session saved to {SESSION_CACHE_FILE}."})


@chat_router.post("/session/restore")
async def restore_session():
    try:
        session_data = torch.load(SESSION_CACHE_FILE)

        ModelState.session_ids = session_data["session_ids"]
        ModelState.cache.current_seq_len = session_data["current_seq_len"]

        for i, (k, v) in enumerate(
            zip(
                session_data["kv_cache"]["key_states"],
                session_data["kv_cache"]["value_states"],
            )
        ):
            ModelState.cache.key_states[i].copy_(
                k.to(ModelState.cache.key_states[i].device)
            )
            ModelState.cache.value_states[i].copy_(
                v.to(ModelState.cache.value_states[i].device)
            )

        ModelState.session_active = True
        return JSONResponse(
            content={"message": f"Session restored from {SESSION_CACHE_FILE}."}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@chat_router.post("/clear")
async def clear_chat():
    chat_context.clear()
    return JSONResponse(content={"message": "Chat context cleared."})


@chat_router.post("/clearall")
async def clear_all():
    ModelState.cache.reset()
    ModelState.session_ids = None
    ModelState.session_active = False
    return JSONResponse(content={"message": "All context cleared."})


@chat_router.post("/instruct")
async def add_instruction(request: ChatRequest):
    instruction = request.prompt.strip()
    print(f"Received instruction: {instruction}")
    instruction_context.append(instruction)
    trim_instruction_context()
    return JSONResponse(content={"message": "Instruction added."})


@chat_router.get("/filetypes")
async def get_filetypes():
    return JSONResponse(content=".txt,.pdf,.zip,.tar.gz")


@chat_router.post("/load")
async def load_state():
    load_checkpoint(ModelState.model, SAFETENSORS_FILE)
    return JSONResponse(content={"message": "Loaded checkpoint."})


@chat_router.post("/save")
async def save_state():
    save_checkpoint(ModelState.model, SAFETENSORS_FILE)
    return JSONResponse(content={"message": "Checkpoint saved."})


@chat_router.post("/upload")
async def upload_stub():
    return JSONResponse(content={"message": "Upload logic not implemented."})


@chat_router.post("/stream")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.prompt), media_type="text/event-stream"
    )


@chat_router.post("/session/merge")
async def merge_session():
    try:
        session_data = torch.load(SESSION_CACHE_FILE)

        loaded_ids = session_data["session_ids"]
        current_ids = ModelState.session_ids

        if current_ids is None:
            return JSONResponse(
                status_code=400, content={"error": "No active session to merge into."}
            )

        # Find matching prefix
        match_len = min(loaded_ids.size(-1), current_ids.size(-1))
        if not torch.equal(loaded_ids[:, :match_len], current_ids[:, :match_len]):
            return JSONResponse(
                status_code=400, content={"error": "Session prefix mismatch."}
            )

        # Load cache tensors to a temp cache on correct device
        loaded_k = session_data["kv_cache"]["key_states"]
        loaded_v = session_data["kv_cache"]["value_states"]

        for i, (k, v) in enumerate(zip(loaded_k, loaded_v)):
            ModelState.cache.key_states[i][:, :match_len].copy_(
                k.to(ModelState.cache.key_states[i].device)
            )
            ModelState.cache.value_states[i][:, :match_len].copy_(
                v.to(ModelState.cache.value_states[i].device)
            )

        ModelState.cache.current_seq_len = match_len

        # Re-feed the tail (if any)
        if current_ids.size(-1) > match_len:

            ModelState.generator._gen_feed_tokens(
                current_ids[:, match_len:], ModelState.settings
            )

        return JSONResponse(
            content={"message": f"Merged cached session up to token {match_len}."}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
