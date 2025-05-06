import librosa
import numpy as np
import scipy

from io import BytesIO
from numpy import ndarray

def detect_onset(audio_buffer:BytesIO)->ndarray:
    y, sr = librosa.load(audio_buffer, sr=None)

    # Onset 검출
    padding_duration = 0.05  # 0.05sec = 50ms
    pad_samples = int(padding_duration * sr)
    y_padded = np.concatenate([np.zeros(pad_samples), y])
    onset_env = librosa.onset.onset_strength(y=y_padded, sr=sr)
    onset_env_smooth = scipy.ndimage.median_filter(onset_env, size=3) #해당 필터로 스무딩
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env_smooth, 
        sr=sr,
        # delta=0.3, #기본값보다 높여서 작은 진폭 무시
        # wait=10, #연속 온셋 사이 최소 프레임 거리
        # pre_max=10, #최대값 검출시 앞뒤로 볼 프레임 수
        # post_max=20
        )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times
