import os
from io import BytesIO

import torch
import librosa
import soundfile as sf

from app.model.inference import CNN_inference
import app.model.utils as utils
from app.model.model import MultiCNN
from .audioToWavConverter import decode_audio_to_wav

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_MEL_TRANSFORM = utils.get_mel_transform("torch")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "drum_model_multiCNN_3_0.25wav_f1_0.91.pt")

# 모델 전역변수 선언
MODEL = MultiCNN().to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,  weights_only=True))
MODEL.eval() #모델을 평가모드로 전환

def predict(segment_audio:BytesIO):
    # 메모리 버퍼에 WAV 데이터 작성
    segment_audio.seek(0)
    pred_label = CNN_inference(DEVICE, TORCH_MEL_TRANSFORM, MODEL, segment_audio)
    return pred_label


def split_audio_and_predict(audio_buffer:BytesIO, onset_times:list):
    """ 한 마디 길이의 오디오를 온셋 단위로 분할 후 모델 예측 수행 """
    margin = 0.00
    shift = 0.05 
    
    wav_buffer = decode_audio_to_wav(audio_buffer=audio_buffer)

    # 전체 오디오 메모리에 로드
    audio_buffer.seek(0)
    y, sr = librosa.load(wav_buffer, sr=None)

    predictions = []
    # 온셋별 구간 분할 및 예측
    for i in range(len(onset_times)):
        # (1) segment 구간 설정(start_time, end_time)
        """예시
        onset_times = [1.0, 2.0, 3.1], 총 길이 4초인 경우
        0~0.9는 버려짐
        0.9~1.9 / 1.9~3.0 / 3.0~3.9 
        """
        start_time = max(0, onset_times[i] - margin - shift)
        if i < len(onset_times) - 1:
            end_time = max(0, onset_times[i+1] - shift)
        else:
            end_time = (len(y) / sr) - shift

        # (2) 실제로 오디오 자르기(최대 0.25초 길이)
        """예시
        start_time, end_time = 3.0, 3.9  ->  3.0~3.25 분할
        start_time, end_time = 3.0, 3.14  ->  3.0~3.14 분할
        """
        target_duration = 0.25

        start_sample = int(start_time * sr)
        end_sample = int(min(end_time, start_time + target_duration) * sr) 
        segment = y[start_sample:end_sample]

        # (3) Numpy 오디오 배열 -> Base64 변환
        buffer = BytesIO()
        sf.write(buffer, segment, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        # (5) 모델 예측
        pred_label = predict(buffer)
        predictions.append(pred_label)

    return predictions

        

