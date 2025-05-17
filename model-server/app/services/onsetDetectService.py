import tempfile
import librosa
import numpy as np
import scipy
from scipy.signal import butter, lfilter

from io import BytesIO
from numpy import ndarray
from .audioToWavConverter import decode_audio_to_wav

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut=500, highcut=6000, fs=44100, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def detect_onset(audio_buffer:BytesIO)->ndarray:
    
    wav_buffer = decode_audio_to_wav(audio_buffer=audio_buffer)
    y, sr = librosa.load(wav_buffer, sr=None)

    # 주파수 필터 적용: 500~6000Hz만 통과
    y_filtered = bandpass_filter(y, lowcut=500, highcut=6000, fs=sr)

    # 앞부분 padding (초반 타격 손실 방지)
    padding_duration = 0.05
    pad_samples = int(padding_duration * sr)
    y_padded = np.concatenate([np.zeros(pad_samples), y_filtered])

    # 온셋 감지용 에너지 계산
    onset_env = librosa.onset.onset_strength(y=y_padded, sr=sr)

    # Median filter로 잡음 smoothing
    onset_env_smooth = scipy.ndimage.median_filter(onset_env, size=3)

    # 온셋 감지
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env_smooth,
        sr=sr,
        backtrack=True,
        pre_max=10,
        post_max=10,
        pre_avg=30,
        post_avg=30,
        delta=0.3,
        wait=10
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times
