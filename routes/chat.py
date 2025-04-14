from datetime import datetime

from auth import ACTIVE_SESSIONS, require_session, verify_jwt_and_create_session
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from model.generation import generate_stream
from model.init import ModelState
from schema import ChatRequest

chat_router = APIRouter()


@chat_router.post("/connect")
async def connect(session=Depends(verify_jwt_and_create_session)):
    """Authenticate via JWT and return a session token."""
    return {
        "message": f"Authenticated as {session['username']}",
        "session_id": session["session_id"],
    }


@chat_router.post("/keepalive")
async def keepalive(session=Depends(require_session)):
    """Ping to keep the session alive."""
    for sid, s in ACTIVE_SESSIONS.items():
        if s is session:
            ACTIVE_SESSIONS[sid]["last_seen"] = datetime.utcnow()
            break
    return {"message": "Keep-alive acknowledged."}


@chat_router.post("/stream")
async def stream_chat(request: ChatRequest, session=Depends(require_session)):
    """Stream a completion from the model."""
    return StreamingResponse(
        generate_stream(request.prompt), media_type="text/event-stream"
    )


@chat_router.post("/clear")
async def clear(session=Depends(require_session)):
    """Reset model cache and clear contexts."""
    ModelState.generator.reset_page_table()
    return JSONResponse(content={"message": "Context cleared."})


@chat_router.post("/clearall")
async def clear_all(session=Depends(require_session)):
    """Reset model cache, session state, and clear contexts."""
    ModelState.cache.reset()
    ModelState.session_ids = None
    ModelState.session_active = False
    return JSONResponse(content={"message": "All context cleared."})


@chat_router.get("/filetypes")
async def get_filetypes():
    """List supported file upload types."""
    return JSONResponse(content=".txt,.pdf,.zip,.tar.gz")
