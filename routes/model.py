import shutil
import subprocess
from pathlib import Path

from auth import require_session
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from model.generation import start_stream
from model.init import ModelState, load_session_into_cache
from model.SupportedModel import (
    SUPPORTED_MODELS,
    SupportedModel,
    get_active_model,
    get_model_by_name,
)

router = APIRouter(prefix="/model")


@router.get("/models")
async def list_supported_models():
    return JSONResponse(
        content={
            "supported_models": str(SupportedModel.get_supported_models()),
        }
    )


@router.post("/set")
async def load_model(request: Request):
    data = await request.json()
    modelName = data.get("model")
    model = get_model_by_name(modelName)
    model.set_model()
    # TODO: Reload the server & add error handling
    print("reloading model")
    subprocess.run(["pkill", "-f", "python server.py"])
    return JSONResponse(
        content={"message": f"Model '{modelName}' loaded. Server will restart now..."}
    )


@router.get("/get")
async def get_model():
    model = get_active_model()
    if model is None:
        return JSONResponse(
            status_code=407,
            content={"message": "Active Model is NONE"},
        )
    return JSONResponse(content={"model": f"'{model.name}'"})
