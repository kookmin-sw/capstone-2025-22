from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.onsetRouter import onsetRouter
from app.routes.predictRouter import predictRouter

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
app.include_router(router=predictRouter)