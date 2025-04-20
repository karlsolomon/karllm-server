import uvicorn
from fastapi import FastAPI

import model
from routes.chat import router as chat_router
from routes.conversation import router as convo_router
from routes.help import router as help_router
from routes.model import router as model_router
from routes.session import router as session_router

app = FastAPI()
app.include_router(chat_router)
app.include_router(session_router)
app.include_router(convo_router)
app.include_router(help_router)
app.include_router(model_router)


@app.on_event("startup")
async def startup_event():
    model.init.load_model()


if __name__ == "__main__":
    uvicorn.run(app, host="10.224.174.3", port=34199, log_level="info")
