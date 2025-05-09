from fastapi import APIRouter, HTTPException
from app.schema.AudioRequest import AudioRequest
from app.schema.AudioResponse import AudioResponse
from app.services.onsetDetectService import detect_onset
import io, base64

onsetRouter = APIRouter(prefix="/onset")

@onsetRouter.post("", tags=["onset"], description="온셋 감지", response_model=AudioResponse)
async def detect_onset_base64(request: AudioRequest)->AudioResponse:
    if not request.audio_base64 :
        raise HTTPException(status_code=400, detail="audio base64 값이 없음")
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        offset_list = detect_onset(audio_buffer=audio_buffer).tolist()  
        return AudioResponse(
            onsets=offset_list,
            count=len(offset_list)
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="base64 처리 중 오류 발생")