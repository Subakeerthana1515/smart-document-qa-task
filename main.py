import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from src import database
from api.sessions import router as sessions_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    database.init_db()
    yield


app = FastAPI(
    title="Smart Document Q&A API",
    description="RAG-based document question answering powered by Gemini 2.5 Flash.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
