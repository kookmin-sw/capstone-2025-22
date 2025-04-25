from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.onsetRouter import onsetRouter

app = FastAPI()

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