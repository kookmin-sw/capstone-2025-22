import os
from io import BytesIO

import torch
import librosa
import soundfile as sf

from model.inference import CNN_inference
import model.utils as utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_MEL_TRANSFORM = utils.get_mel_transform("torch")
MODEL_PATH = "../model/drum_model_multiCNN_20_0.25wav_f1_0.92.pt"

def predict(segment_audio:BytesIO):
    # 메모리 버퍼에 WAV 데이터 작성
    segment_audio.seek(0)
    label = CNN_inference(DEVICE, TORCH_MEL_TRANSFORM, MODEL_PATH, segment_audio)
    return label


def split_audio_and_predict(audio_buffer:BytesIO, onset_times:list):
    """ 한 마디 길이의 오디오를 온셋 단위로 분할 후 모델 예측 수행 """
    margin = 0.00
    shift = 0.05 
    
    # 전체 오디오 메모리에 로드
    audio_buffer.seek(0)
    y, sr = librosa.load(audio_buffer, sr=None)

    predictions = []
    # 온셋별 구간 분할 및 예측
    for idx, t in enumerate(onset_times):
        start_time = max(0.0, t - margin - shift)
        if idx < len(onset_times) - 1:
            end_time = max(0.0, onset_times[idx + 1] - shift)
        else:
            end_time = len(y) / sr - shift

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

        

