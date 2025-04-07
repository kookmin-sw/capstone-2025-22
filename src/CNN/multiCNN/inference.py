import torch
import torchaudio
import os, sys
from model import MultiCNN

previous_folder = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(previous_folder)

from utils import convert_to_wav

def predict_drum_sound(device, mel_transform, model_path, wav_path, sample_rate=44100):
    class_labels = ["kick", "snare", "overheads", "toms"]

    # 모델 로드
    model = MultiCNN(num_classes=len(class_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() #모델을 평가모드로 전환

    # 오디오 로드
    waveform, sr = torchaudio.load(wav_path)
    # print("Original waveform shape:", waveform.shape)

    if sr != sample_rate:
        # 필요하면 리샘플링
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        # print("Resampled waveform shape:", waveform.shape)

    if waveform.shape[0] != 1: #1채널로 변환
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print("After mono conversion shape:", waveform.shape)

    #MelSpectrogram 변환
    features = mel_transform(waveform)
    # print("After mel_transform shape:", features.shape)

    # 모델에 입력 전에 배치 차원 추가 (확인용)
    if len(features.shape) == 3:  # [채널, 높이, 너비] 형태라면
        features = features.unsqueeze(0)  # [1, 채널, 높이, 너비]로 변환
        # print("After adding batch dimension:", features.shape)

    # 디바이스로 이동
    features = features.to(device)

    with torch.no_grad(): #학습을 위한 기울기 계산 x
        outputs = model(features) #입력 데이터를 모델에 통과시켜 예측값 얻기
        _, pred = torch.max(outputs, dim=1) #예측 확률 중 가장 높은 값 반환
    
    predicted_label = class_labels[pred]

    return predicted_label

def CNN_inference(device, mel_transform, model_path, test_data_folder_path):
    # 폴더 내 모든 wav 파일 예측 수행
    for test_audio in os.listdir(test_data_folder_path):
        test_audio_path = os.path.join(test_data_folder_path, test_audio)
        if test_audio.startswith('.') or os.path.isdir(test_audio_path): continue # .DS_Store 파일 및 다른 숨김 파일 건너뛰기
        _, filetype = os.path.splitext(test_audio)
        if filetype != ".wav":
            test_audio_path = convert_to_wav(test_audio)
        
        pred_label = predict_drum_sound(device, mel_transform, model_path, test_audio_path)
        print(f"Predicted Drum Sound: {test_audio} -> [{pred_label}]")

