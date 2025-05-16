import tempfile
import librosa
import numpy as np
import scipy

from io import BytesIO
from numpy import ndarray
from .audioToWavConverter import decode_audio_to_wav

def detect_onset(audio_buffer:BytesIO)->ndarray:
    
    wav_buffer = decode_audio_to_wav(audio_buffer=audio_buffer)
    y, sr = librosa.load(wav_buffer, sr=None)

    padding_duration = 0.05  # 0.05sec = 50ms
    pad_samples = int(padding_duration * sr)
    y_padded = np.concatenate([np.zeros(pad_samples), y])
    onset_env = librosa.onset.onset_strength(y=y_padded, sr=sr)
    onset_env_smooth = scipy.ndimage.median_filter(onset_env, size=3) #해당 필터로 스무딩
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env_smooth, 
        sr=sr,
        )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times
