import torch
import torch.nn as nn

import app.model.utils as utils

MAX_FRAMES = 86 #0.25초 기준 sr=44100, hop=128일 때의 대략적인 멜 스펙토그램의 프레임 수
class_labels = ["KD", "SD", "CY", "TT", "HH"]  # 5개 레이블

def predict_drum_sound(model, wav_file, mel_transform, device):
    # 1. base64 오디오 -> 멜스펙트로그램으로 변환
    features = utils.wav_to_mel(wav_file, mel_transform) #ex. [1=channel, 128=n_mels, T=time_frame]

    # 2. 고정 크기 변환
    pool = nn.AdaptiveAvgPool2d((128, MAX_FRAMES)).to(device)
    features = pool(features)

    # 3. 모델 입력 전, 배치 차원 추가 (학습 때는 DataLoader에서 해줌)
    features = features.unsqueeze(0).to(device)  # [1, 채널, 높이, 너비]로 변환

    # 4. 모델 추론
    with torch.no_grad():
        outputs = model(features)
        predictions = utils.select_one_or_two_classes(outputs, debug=False)
    
    # 5. 추론 결과
    predicted_labels = [] #이진 표기 예측에서 레이블 추출
    for i, pred in enumerate(predictions[0]):
        if pred.item() == 1.0:
            predicted_labels.append(class_labels[i])

    return predicted_labels

def CNN_inference(device, mel_transform, model, segment_audio):
    # test_audio_path = utils.wav_file_to_base64(segment_audio) #wav_path -> base64 문자열로 변환
    pred_label = predict_drum_sound(model, segment_audio, mel_transform, device)

    print(f"Predicted Drum Sound: [{pred_label}]")
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    return pred_label

