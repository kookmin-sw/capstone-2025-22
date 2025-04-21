import os
from pydub import AudioSegment
import torchaudio
import torch

import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import torch.nn.functional as F

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
    
def wav_to_mel(wav_path, mel_transform, sample_rate=44100):
    waveform, sr = torchaudio.load(wav_path)
    # print("Original waveform shape:", waveform.shape)

    if sr != sample_rate:
        # 필요하면 리샘플링
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        # print("Resampled waveform shape:", waveform.shape)

    if waveform.shape[0] > 1: #1채널로 변환
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print("After mono conversion shape:", waveform.shape)
    
    #MelSpectrogram 변환
    features = mel_transform(waveform)
    # print("After mel_transform shape:", features.shape)
    
    return features


def select_one_or_two_classes(logits):
    thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]) #KD, SD, CY, TT, HH 각 클래스별 최적 임계치
    probs = torch.sigmoid(logits)                   # 각 클래스별 확률을 0~1 사이로 변환 (ex. [[0.2, 0.6, 0.4, 0.8, 0.1]])
    thresh = thresholds.view(1, -1).to(probs.device)
    preds = (probs > thresh).float()                # 클래스별 임계치 적용 (ex. [[0, 1, 0, 1, 0]])

    print(probs)
    print(preds)
    # 최소 1개, 최대 2개 보정
    for i in range(preds.size(0)):
        cnt = int(preds[i].sum().item())
        if cnt == 0: # 모두 0이면, 가장 확률 높은 클랙스 하나 선택
            idx = probs[i].argmax().item()
            preds[i, idx] = 1.0
        elif cnt > 2: # 3개 이상이면, 가장 확률 높은 2개 선택
            top2 = probs[i].topk(2).indices
            mask = torch.zeros_like(preds[i])
            mask[top2] = 1.0
            preds[i] = mask
    return preds
  

def visualize_mel_spectrogram(wav_path):
    """
    오디오 파일을 PyTorch 멜 스펙트로그램으로 변환하고 시각화합니다.
    """
    # 오디오 데이터 로드
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:  # 모노로 변환
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # PyTorch 멜 스펙트로그램 생성
    torch_mel_transform = get_mel_transform("torch")
    mel_spec_torch = torch_mel_transform(waveform)
    mel_spec_torch_np = mel_spec_torch.squeeze(0).numpy()
    mel_spec_db_torch = librosa.power_to_db(mel_spec_torch_np, ref=np.max)

    # 시각화 (PyTorch 멜 스펙트로그램만)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db_torch,
        y_axis='mel',
        x_axis='time',
        sr=sr
    )
    plt.title('PyTorch Mel Spectrogram')
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    # 멜 스펙트로그램 형태 정보 출력
    print(f"PyTorch Mel Spectrogram: {mel_spec_torch.shape} - 채널 {mel_spec_torch.shape[0]}, 특성 {mel_spec_torch.shape[1]}, 시간 프레임 {mel_spec_torch.shape[2]}")
    print("KD, SD, CY, TT, HH")
    return mel_spec_torch

