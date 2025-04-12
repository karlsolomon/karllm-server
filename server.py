import uvicorn
from fastapi import FastAPI

from model.init import lazy_load_model
from routes.chat import chat_router

app = FastAPI()
app.include_router(chat_router)


@app.on_event("startup")
async def startup_event():
    lazy_load_model()


if __name__ == "__main__":
    uvicorn.run(app, host="10.0.0.90", port=34199, log_level="info")
