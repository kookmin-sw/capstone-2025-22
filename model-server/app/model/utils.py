import io
import base64
import soundfile as sf
import torchaudio
import torch

import librosa
import torch.nn.functional as F

def get_mel_transform(library, filepath=None):
    sample_rate=44100
    n_fft=1024
    hop_length=128
    n_mels=64

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

    # 1. 오디오 로드
    if isinstance(wav_file, str):
        # base64 문자열로 처리
        try:
            decoded = base64.b64decode(wav_file)
            buffer = io.BytesIO(decoded)
            waveform, sr = sf.read(buffer) # [Time, Channel]
            # [Channel, Time]로 바꿔주고, float32타입으로 변환해줘야 모델이 인식함
            if waveform.ndim == 1:  # 모노
                waveform = torch.tensor(waveform).unsqueeze(0).float()
            else: 
                waveform = torch.tensor(waveform).T.float() # 스테레오
        except Exception as e:
            raise ValueError("base64 decoding 실패 또는 잘못된 형식입니다.") from e
    elif isinstance(wav_file, io.BytesIO):
        # BytesIO 객체 직접 처리
        try:
            waveform, sr = sf.read(wav_file) # [Time, Channel]
            if waveform.ndim == 1:  # 모노
                waveform = torch.tensor(waveform).unsqueeze(0).float()
            else: 
                waveform = torch.tensor(waveform).T.float() # 스테레오
        except Exception as e:
            raise ValueError("BytesIO 객체 처리 중 오류 발생") from e
    else:
        raise TypeError("wav_file은 base64 문자열 또는 BytesIO 객체여야 합니다.")

    # 샘플레이트 맞추기
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # 모노 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print("After mono conversion shape:", waveform.shape)

    # 4. 길이 고정 (0.25초)
    target_samples = int(44100 * 0.25)

    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]  # 앞에서 자르기
    elif waveform.shape[1] < target_samples:
        pad_amount = target_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_amount))  # 뒤쪽 padding
    
    # 멜 스펙트로그램으로 변환
    features = mel_transform(waveform)
    # print("After mel_transform shape:", features.shape)
    return features



def select_one_or_two_classes(logits, debug=False):
    thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]) #KD, SD, CY, TT, HH 각 클래스별 최적 임계치
    probs = torch.sigmoid(logits)                   # 각 클래스별 확률을 0~1 사이로 변환 (ex. [[0.2, 0.6, 0.4, 0.8, 0.1]])
    thresh = thresholds.view(1, -1).to(probs.device)
    preds = (probs > thresh).float()                # 클래스별 임계치 적용 (ex. [[0, 1, 0, 1, 0]])

    if debug:
        prob_list = probs[0].detach().cpu().numpy().round(3)
        pred_list = preds[0].detach().cpu().numpy().astype(int)
        print("probs:", ", ".join(f"{p:.3f}" for p in prob_list))
        print("preds:", ", ".join(str(p) for p in pred_list))
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
  