from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.routes.onsetRouter import onsetRouter
from app.routes.predictRouter import predictRouter
from app.services.onsetDetectService import detect_onset

from pathlib import Path
from io import BytesIO

@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_path = Path(__file__).parent / "resources" / "warmup.wav"
    with open(warmup_path, "rb") as f:
        dummy_audio = BytesIO(f.read())
    detect_onset(dummy_audio)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    print("hello world success")
    return {"message": "model server started"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=onsetRouter)
app.include_router(router=predictRouter)