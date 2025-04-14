from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

from .conversation import router as convo_router  # Optional, for introspection
from .session import router as session_router  # Optional, for introspection

HELP_METADATA = [
    {
        "path": "/connect",
        "method": "POST",
        "description": "Authenticate client and start a new session",
        "body": {
            "saveInteractions": "bool (optional): Whether to save .safetensors files during session"
        },
    },
    {
        "path": "/keepalive",
        "method": "POST",
        "description": "Keep session alive to avoid timeout",
    },
    {
        "path": "/stream",
        "method": "POST",
        "description": "Stream model response for a given prompt",
        "body": {"prompt": "str: Input text prompt"},
    },
    {
        "path": "/clear",
        "method": "POST",
        "description": "Clear current conversation context (but preserve session)",
    },
    {
        "path": "/clearall",
        "method": "POST",
        "description": "Reset full model context and session state",
    },
    {
        "path": "/filetypes",
        "method": "GET",
        "description": "List supported file upload types",
    },
    {
        "path": "/convo/eraseHistory",
        "method": "POST",
        "description": "Delete all old session folders except the current one",
    },
    {
        "path": "/convo/list",
        "method": "GET",
        "description": "List saved session folders for the user",
    },
    {
        "path": "/convo/load/{name}",
        "method": "POST",
        "description": "Restore a specific conversation session by name",
    },
    {
        "path": "/convo/delete/{name}",
        "method": "POST",
        "description": "Delete a specific session folder (not the active one)",
    },
    {
        "path": "/convo/rename/{old}/{new}",
        "method": "POST",
        "description": "Rename a saved conversation session",
    },
    {
        "path": "/help",
        "method": "GET",
        "description": "Show a table of available API endpoints",
    },
    {
        "path": "/help/json",
        "method": "GET",
        "description": "Return structured metadata describing the API (this endpoint)",
    },
]


@router.get("/help")
async def help_menu():
    header = "| Method | Endpoint | Description |\n|--------|----------|-------------|"
    rows = [
        f"| {e['method']} | `{e['path']}` | {e['description']} |" for e in HELP_METADATA
    ]
    return JSONResponse(content={"help": "\n".join([header] + rows)})


@router.get("/help/json")
async def help_metadata():
    return JSONResponse(content={"endpoints": HELP_METADATA})
