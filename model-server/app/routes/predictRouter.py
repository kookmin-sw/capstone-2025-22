# routers/predictRouter.py
from fastapi import APIRouter, HTTPException
from schema import PredictRequest, PredictResponse
from services.predictDrumService import split_audio_and_predict
import io, base64

predictRouter = APIRouter(prefix="/onset/predict")

@predictRouter.post(
    "",
    tags=["predict"],
    description="온셋 단위 Drum 예측",
    response_model=PredictResponse
)
async def predict_drum_base64(request: PredictRequest) -> PredictResponse:
    if not request.audio_base64 or not request.onsets:
        raise HTTPException(
            status_code=400,
            detail="audio base64 혹은 onsets 정보가 없습니다"
        )
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        predictions = split_audio_and_predict(audio_buffer, request.onsets)
        return PredictResponse(predictions=predictions)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="예측 처리 중 오류 발생")
