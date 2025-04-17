import shutil
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from auth import require_session
from model.generation import start_stream
from model.init import ModelState, load_session_into_cache
from model.SupportedModel import SUPPORTED_MODELS, SupportedModel, get_model_by_name

router = APIRouter(prefix="/model")


@router.get("/list")
async def list_supported_models(session=Depends(require_session)):
    return JSONResponse(
        content={
            "supported_models": str(SupportedModel.get_supported_models()),
        }
    )


@router.post("/set/{name}")
async def load_model(name: str, session=Depends(require_session)):
    model = get_model_by_name(name)
    if model is None:
        return JSONResponse(
            status_code=406,
            content={"message": f"Model '{name}' not supported."},
        )
    model.set_model()
    # TODO: Reload the server & add error handling
    return JSONResponse(
        content={"message": f"Model '{name}' loaded. Server will restart now..."}
    )


@router.get("/get")
async def get_model(session=Depends(require_session)):
    model = SupportedModel.get_active_model()
    if model is None:
        return JSONResponse(
            status_code=407,
            content={"message": "Active Model is NONE"},
        )
    return JSONResponse(content={"model": f"'{model.name}'"})
