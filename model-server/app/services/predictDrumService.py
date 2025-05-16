import os
import tempfile
from io import BytesIO

import torch
import librosa
import soundfile as sf

from app.model.inference import CNN_inference
import app.model.utils as utils

from .audioToWavConverter import decode_audio_to_wav

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_MEL_TRANSFORM = utils.get_mel_transform("torch")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "drum_model_multiCNN_20_0.25wav_f1_0.92.pt")


def predict(segment_audio:BytesIO):
    # 메모리 버퍼에 WAV 데이터 작성
    segment_audio.seek(0)
    label = CNN_inference(DEVICE, TORCH_MEL_TRANSFORM, MODEL_PATH, segment_audio)
    return label


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
    for idx, t in enumerate(onset_times):
        start_time = max(0.0, t - margin - shift)
        end_time = start_time + 0.25

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        # numpy 배열을 WAV BytesIO로 변환
        buffer = BytesIO()
        sf.write(buffer, segment, sr, format='WAV')
        buffer.seek(0)

        # 모델 예측
        pred_label = predict(buffer)
        predictions.append(pred_label)

    return predictions

        

