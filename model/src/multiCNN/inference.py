import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import utils
from model import MultiCNN

MAX_FRAMES = 86

def predict_drum_sound(device, mel_transform, model_path, wav_file, pool):
    class_labels = ["KD", "SD", "CY", "TT", "HH"]  # 5개 레이블
    
    # 모델 로드
    model = MultiCNN().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval() #모델을 평가모드로 전환

    # 오디오 파일(바이트 데이터 또는 파일 경로)을 멜스펙트로그램으로 변환
    features = utils.wav_to_mel(wav_file, mel_transform) #ex. [1, 128=n_mels, T]
    # 고정 크기 변환
    _, _, T = features.shape
    if T < MAX_FRAMES:
        pad_amount = MAX_FRAMES - T
        features = F.pad(features, (0, pad_amount))
    
    features = pool(features)

    # 모델에 입력 전에 배치 차원 추가 (학습 때는 DataLoader에서 해줌)
    if len(features.shape) == 3:  # [채널, 높이, 너비] 형태라면
        features = features.unsqueeze(0)  # [1, 채널, 높이, 너비]로 변환

    features = features.to(device)

    with torch.no_grad(): #학습을 위한 기울기 계산 x
        outputs = model(features) #입력 데이터를 모델에 통과시켜 예측값 얻기
        predictions = utils.select_one_or_two_classes(outputs)
    
    predicted_labels = [] #이진 표기 예측에서 레이블 추출
    for i, pred in enumerate(predictions[0]):
        if pred.item() == 1.0:
            predicted_labels.append(class_labels[i])

    return predicted_labels

def CNN_inference(device, mel_transform, model_path):
    pool = nn.AdaptiveAvgPool2d((128, MAX_FRAMES)).to(device)

    test_data_folder_path = "../test_data"
    for test_audio in os.listdir(test_data_folder_path):
        test_audio_path = os.path.join(test_data_folder_path, test_audio)
        if test_audio.startswith('.') or os.path.isdir(test_audio_path): continue # .DS_Store 파일 및 다른 숨김 파일 건너뛰기
        _, filetype = os.path.splitext(test_audio)
        if filetype != ".wav":
            test_audio_path = utils.convert_to_wav(test_audio)
        
        #wav_file -> base64 문자열로 변환
        test_audio_path = utils.wav_file_to_base64(test_audio_path)
        
        pred_label = predict_drum_sound(device, mel_transform, model_path, test_audio_path, pool)

        print(f"Predicted Drum Sound: {test_audio} -> [{pred_label}]")
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

