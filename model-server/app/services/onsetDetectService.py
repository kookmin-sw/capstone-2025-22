import tempfile
import librosa
import numpy as np
import scipy

from io import BytesIO
from numpy import ndarray

def detect_onset(audio_buffer:BytesIO)->ndarray:
    # y, sr = librosa.load(audio_buffer, sr=None)

    # padding_duration = 0.05  # 0.05sec = 50ms
    # pad_samples = int(padding_duration * sr)
    # y_padded = np.concatenate([np.zeros(pad_samples), y])
    # onset_env = librosa.onset.onset_strength(y=y_padded, sr=sr)
    # onset_env_smooth = scipy.ndimage.median_filter(onset_env, size=3) #해당 필터로 스무딩
    # onset_frames = librosa.onset.onset_detect(
    #     onset_envelope=onset_env_smooth, 
    #     sr=sr,
    #     )
    # onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # return onset_times
    # 1. 임시 파일에 저장
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=True) as tmp:
        tmp.write(audio_buffer.getbuffer())
        tmp.flush()  # Ensure it's written

        # 2. librosa로 로드
        y, sr = librosa.load(tmp.name, sr=None)

    # 3. onset 검출 로직
    padding_duration = 0.05
    pad_samples = int(padding_duration * sr)
    y_padded = np.concatenate([np.zeros(pad_samples), y])
    onset_env = librosa.onset.onset_strength(y=y_padded, sr=sr)
    onset_env_smooth = scipy.ndimage.median_filter(onset_env, size=3)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env_smooth, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times
