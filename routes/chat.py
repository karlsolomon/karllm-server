from auth import require_session
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from model.generation import continue_prompt
from schema import ChatRequest

router = APIRouter()


@router.post("/stream")
async def stream_chat(request: ChatRequest, session=Depends(require_session)):
    return StreamingResponse(
        continue_prompt(request.prompt), media_type="text/event-stream"
    )
