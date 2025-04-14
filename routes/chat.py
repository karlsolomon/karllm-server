import shutil
from datetime import datetime
from pathlib import Path

import config
from auth import ACTIVE_SESSIONS, require_session, verify_jwt_and_create_session
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from model.generation import continue_prompt, start_stream
from model.init import ModelState
from schema import ChatRequest

chat_router = APIRouter()


@chat_router.post("/connect")
async def connect(request: Request, session=Depends(verify_jwt_and_create_session)):
    username = session["username"]

    # Read client config values (e.g. saveInteractions) from POST body
    body = await request.json()
    save_interactions = body.get("saveInteractions", False)
    ModelState.save_interactions = save_interactions

    # Attach to session in ACTIVE_SESSIONS
    session_id = session["session_id"]
    if session_id in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id]["saveInteractions"] = save_interactions

    # Create session directory
    # Define the base session directory using the authenticated username.
    base_sessions_dir = Path(config.SESSION_DIR) / username / "sessions"
    base_sessions_dir.mkdir(parents=True, exist_ok=True)

    # List existing session directories (only those with numeric names) to calculate the new session number.
    existing_sessions = [
        int(d.name)
        for d in base_sessions_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]
    new_session_number = max(existing_sessions) + 1 if existing_sessions else 0

    # Create the new session directory.
    new_session_dir = base_sessions_dir / str(new_session_number)
    new_session_dir.mkdir()

    ModelState.session_dir = new_session_dir

    # Initialize streaming (or other per-session setups) as needed.
    start_stream()

    return {
        "message": f"Authenticated as {username}",
        "session_id": session["session_id"],
        "session_directory": str(new_session_dir),
    }


@chat_router.post("/keepalive")
async def keepalive(session=Depends(require_session)):
    for sid, s in ACTIVE_SESSIONS.items():
        if s is session:
            ACTIVE_SESSIONS[sid]["last_seen"] = datetime.utcnow()
            break
    return {"message": "Keep-alive acknowledged."}


@chat_router.post("/stream")
async def stream_chat(request: ChatRequest, session=Depends(require_session)):
    """Stream a completion from the model."""
    return StreamingResponse(
        continue_prompt(request.prompt), media_type="text/event-stream"
    )


@chat_router.post("/clear")
async def clear(session=Depends(require_session)):
    """Reset model cache and clear contexts."""
    start_stream()
    # TODO: restore instructions
    return JSONResponse(content={"message": "Context cleared."})


@chat_router.post("/clearall")
async def clear_all(session=Depends(require_session)):
    """Reset model cache, session state, and clear contexts."""
    start_stream()
    return JSONResponse(content={"message": "All context cleared."})


@chat_router.get("/filetypes")
async def get_filetypes(session=Depends(require_session)):
    """List supported file upload types."""
    return JSONResponse(content=".txt,.pdf,.zip,.tar.gz")


@chat_router.post("/convo/eraseHistory")
async def erase_conversation_history(session=Depends(require_session)):
    """Erase all *old* conversation session folders for this user (preserve current session)."""
    username = session["username"]
    sessions_dir = Path("users") / username / "sessions"

    current_session_dir = getattr(ModelState, "session_dir", None)
    deleted = 0

    if not sessions_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"message": f"No session directory found for user '{username}'."},
        )

    for subdir in sessions_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            if (
                current_session_dir
                and subdir.resolve() == Path(current_session_dir).resolve()
            ):
                continue  # Skip the currently active session directory
            shutil.rmtree(subdir)
            deleted += 1

    return JSONResponse(
        content={
            "message": f"Deleted {deleted} old session folder(s) for user '{username}'."
        }
    )
