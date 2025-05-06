from pathlib import Path

from auth import require_session
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from schema import ChatRequest

from model.generation import continue_prompt, encode_file

router = APIRouter(prefix="/chat")


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    return StreamingResponse(
        continue_prompt(req.prompt), media_type="text/event-stream"
    )


@router.post("/upload")
async def upload_file(
    session=Depends(require_session),
    file: UploadFile = File(...),
    newfilename: str = Form(None),
):
    username = session["username"]
    user_dir = Path(f"/home/ksolomon/git/karllm/karllm-server/users/{username}/files")
    user_dir.mkdir(parents=True, exist_ok=True)

    target_name = newfilename or file.filename
    target_path = user_dir / target_name

    with open(target_path, "wb") as f:
        f.write(await file.read())

    return {"message": f"File uploaded as {target_name}"}


@router.post("/read")
async def read_file(data: dict, session=Depends(require_session)):
    filename = data.get("filename")
    username = session["username"]

    file_path = Path(
        f"/home/ksolomon/git/karllm/karllm-server/users/{username}/files/{filename}"
    )
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    injected_tokens = encode_file(file_path)

    return {"message": f"Injected {injected_tokens} tokens into context window."}
