import os
from pydub import AudioSegment
import torchaudio
import torch

import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def convert_to_wav(path):
    _, filetype = os.path.splitext(path)
    if filetype == ".m4a":
        audio = AudioSegment.from_file(path, format="m4a")
        path = path.replace("m4a", "wav")
        audio.export(path, format="wav")
    return path

def get_mel_transform(library, filepath=None):
    sample_rate=44100
    n_fft=2048
    hop_length=512 
    n_mels=128

    if library == "torch":
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    elif library == "librosa":
        audio_data, sr = librosa.load(filepath, sr=None)
        return librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    
def create_combined_plots(filepath):
    audio_data, sr = librosa.load(filepath, sr=None)

    # 멜 스펙트로그램 생성
    mel_spec = get_mel_transform("librosa", filepath)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    # ax1 : 원본 웨이브폼
    times_orig = np.arange(len(audio_data)) / sr
    ax1.plot(times_orig, audio_data)
    ax1.set_title('Original Waveform')
    ax1.set_xlabel('Time (s)')
    
    # ax2 : 멜 스펙트로그램
    img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', ax=ax2)
    ax2.set_title('Mel Spectrogram')
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    combined_image = plt.imread(buf)

    #이미지 출력
    # plt.figure(figsize=(15, 4))
    # plt.imshow(combined_image)
    # plt.axis('off')  # 축 숨기기
    # plt.show()
    #
    return torch.from_numpy(combined_image).float() # NumPy 배열인 combined_image를 PyTorch 텐서로 변환