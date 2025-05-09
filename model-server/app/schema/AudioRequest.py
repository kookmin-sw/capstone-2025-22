from pydantic import BaseModel

class AudioRequest(BaseModel):
    audio_base64: str

class PredictRequest(BaseModel):
    audio_base64: str
    onsets: list[float]