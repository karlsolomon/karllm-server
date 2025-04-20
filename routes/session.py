from datetime import datetime
from pathlib import Path

import config
from auth import ACTIVE_SESSIONS, require_session, verify_jwt_and_create_session
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from model.generation import start_stream
from model.init import ModelState

router = APIRouter()


@router.post("/connect")
async def connect(request: Request, session=Depends(verify_jwt_and_create_session)):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid or empty body")
    except Exception as e:
        print(f"❌ Failed to parse request body: {e}")
        raise HTTPException(status_code=400, detail="Missing or malformed JSON body")
    username = session["username"]
    body = await request.json()
    save_interactions = body.get("saveInteractions", False)
    ModelState.save_interactions = save_interactions
    session_id = session["session_id"]

    if session_id in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id]["saveInteractions"] = save_interactions

    base_sessions_dir = Path(config.SESSION_DIR) / username / "sessions"
    base_sessions_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name)
        for d in base_sessions_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]
    new_id = max(existing) + 1 if existing else 0
    session_dir = base_sessions_dir / str(new_id)
    session_dir.mkdir()

    ModelState.session_dir = session_dir
    start_stream()

    return {
        "message": f"Authenticated as {username}",
        "session_id": session_id,
        "session_directory": str(session_dir),
    }


@router.post("/keepalive")
async def keepalive(session=Depends(require_session)):
    for sid, s in ACTIVE_SESSIONS.items():
        if s is session:
            ACTIVE_SESSIONS[sid]["last_seen"] = datetime.utcnow()
            break
    return {"message": "Keep-alive acknowledged."}


@router.post("/clear")
async def clear(session=Depends(require_session)):
    # TODO:
    start_stream()
    return JSONResponse(content={"message": "Context cleared."})


@router.post("/clearall")
async def clear_all(session=Depends(require_session)):
    start_stream()
    return JSONResponse(content={"message": "All context cleared."})
