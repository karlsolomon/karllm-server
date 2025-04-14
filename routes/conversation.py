import shutil
from pathlib import Path

from auth import require_session
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from model.init import ModelState

router = APIRouter(prefix="/convo")


@router.post("/eraseHistory")
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


@router.get("/list")
async def list_conversations(session=Depends(require_session)):
    """List all saved conversation directories for the current user."""
    username = session["username"]
    sessions_dir = Path("users") / username / "sessions"

    if not sessions_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"message": f"No sessions found for user '{username}'."},
        )

    session_folders = [
        d.name
        for d in sorted(sessions_dir.iterdir())
        if d.is_dir() and (d.name.isdigit() or d.name)
    ]

    return JSONResponse(
        content={
            "username": username,
            "conversations": session_folders,
        }
    )


@router.post("/load/{name}")
async def load_conversation(name: str, session=Depends(require_session)):
    """Set an old session directory as the current active one."""
    username = session["username"]
    target_dir = Path("users") / username / "sessions" / name

    if not target_dir.exists() or not target_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Session '{name}' does not exist for user '{username}'."
            },
        )

    ModelState.session_dir = target_dir
    # TODO: clear cache and load in old cache
    return JSONResponse(
        content={"message": f"Session '{name}' loaded for user '{username}'."}
    )


@router.post("/delete/{name}")
async def delete_conversation(name: str, session=Depends(require_session)):
    """Delete a specific session directory (if not the current one)."""
    username = session["username"]
    target_dir = Path("users") / username / "sessions" / name

    if not target_dir.exists() or not target_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Session '{name}' does not exist for user '{username}'."
            },
        )

    current_session_dir = getattr(ModelState, "session_dir", None)
    if (
        current_session_dir
        and target_dir.resolve() == Path(current_session_dir).resolve()
    ):
        return JSONResponse(
            status_code=403,
            content={"message": f"Cannot delete active session '{name}'."},
        )

    shutil.rmtree(target_dir)
    return JSONResponse(
        content={"message": f"Session '{name}' deleted for user '{username}'."}
    )


@router.post("/rename/{old}/{new}")
async def rename_conversation(old: str, new: str, session=Depends(require_session)):
    """Rename an existing conversation session."""
    username = session["username"]
    sessions_dir = Path("users") / username / "sessions"
    old_path = sessions_dir / old
    new_path = sessions_dir / new

    # Validate existence
    if not old_path.exists() or not old_path.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Session '{old}' does not exist for user '{username}'."
            },
        )

    # Prevent renaming current session
    current_session_dir = getattr(ModelState, "session_dir", None)
    if (
        current_session_dir
        and old_path.resolve() == Path(current_session_dir).resolve()
    ):
        return JSONResponse(
            status_code=403,
            content={"message": f"Cannot rename the currently active session '{old}'."},
        )

    # Prevent overwriting another session
    if new_path.exists():
        return JSONResponse(
            status_code=409,
            content={"message": f"Session name '{new}' already exists."},
        )

    old_path.rename(new_path)
    return JSONResponse(
        content={
            "message": f"Session '{old}' renamed to '{new}' for user '{username}'."
        }
    )
