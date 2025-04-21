import os
import io
import base64
from pydub import AudioSegment
import soundfile as sf
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

def wav_file_to_base64(wav_path):
    # 테스트용(wav_file -> base64 문자열로 변환)
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return encoded
    
def wav_to_mel(wav_file, mel_transform, sample_rate=44100):
    ''' wav_file(base64 문자열 또는 파일 경로) -> 멜 스펙트로그램 '''
    waveform = None
    sr = None

    if isinstance(wav_file, str):
        # (1) .wav로 끝나면 파일 경로라고 간주
        if wav_file.lower().endswith(".wav"):
            print("wav_file is a file path")
            if not os.path.exists(wav_file):
                raise FileNotFoundError(f"'{wav_file}' 경로에 파일이 없습니다.")
            waveform, sr = torchaudio.load(wav_file)

        else: # (2) 그 외에는 base64 문자열이라고 간주
            print("wav_file is a base64 string")
            try:
                decoded = base64.b64decode(wav_file)
                buffer = io.BytesIO(decoded)
                waveform, sr = sf.read(buffer) # [Time, Channel]
                # [Channel, Time]로 바꿔주고, float32타입으로 변환해줘야 모델이 인식함
                if waveform.ndim == 1:  # 모노
                    waveform = torch.tensor(waveform).unsqueeze(0).float()
                else: waveform = torch.tensor(waveform).T.float() # 스테레오
            except Exception as e:
                raise ValueError("base64 decoding 실패 또는 잘못된 형식입니다.") from e
    else:
        raise TypeError("wav_file은 base64 문자열 또는 파일 경로여야 합니다.")
    # print("Original waveform shape:", waveform.shape)

    # 샘플레이트 맞추기
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # 모노 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print("After mono conversion shape:", waveform.shape)
    
    # 멜 스펙트로그램으로 변환
    features = mel_transform(waveform)
    # print("After mel_transform shape:", features.shape)
    return features



def select_one_or_two_classes(logits):
    thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]) #KD, SD, CY, TT, HH 각 클래스별 최적 임계치
    probs = torch.sigmoid(logits)                   # 각 클래스별 확률을 0~1 사이로 변환 (ex. [[0.2, 0.6, 0.4, 0.8, 0.1]])
    thresh = thresholds.view(1, -1).to(probs.device)
    preds = (probs > thresh).float()                # 클래스별 임계치 적용 (ex. [[0, 1, 0, 1, 0]])

    # print(probs)
    # print(preds)
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

