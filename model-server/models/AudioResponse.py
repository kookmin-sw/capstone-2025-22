from pydantic import BaseModel

class AudioResponse(BaseModel):
    onsets: list
    count: int