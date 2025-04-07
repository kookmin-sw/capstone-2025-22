import torch
import torchaudio
import os
from utils import convert_to_wav, create_combined_plots

from model import SimpleCNN, CombinedCNN
from dataset import DrumDataset  # class_to_idx를 재사용할 수도 있음

def predict_drum_sound(device, mel_transform, model_path, wav_path, sample_rate=44100):
    class_labels = ["kick", "snare", "overheads", "toms"]

    # 모델 로드
    model = CombinedCNN(num_classes=len(class_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() #모델을 평가모드로 전환

    # 오디오 로드
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        # 필요하면 리샘플링
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] != 1: #1채널로 변환
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    #MelSpectrogram 변환
    combined_image = create_combined_plots(wav_path)

    # png 이미지(채널4) -> 그레이스케일(채널1)로 변환
    grayscale = combined_image.mean(dim=2)  # [높이, 너비]
    features = grayscale.unsqueeze(0).unsqueeze(0)  # [1, 1, 높이, 너비]
    features = features.to(device)#데이터를 cpu/gpu로 이동

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

