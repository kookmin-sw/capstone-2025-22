from pydantic import BaseModel

class AudioRequest(BaseModel):
    audio_base64: str