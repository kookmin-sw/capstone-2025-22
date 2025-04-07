import os
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np

from utils import create_combined_plots

class DrumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform

        classes = sorted(os.listdir(data_dir)) #클래스 라벨, 폴더명으로부터 추출 (kick, overheads, snare, toms)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # 각 클래스 폴더별 wav 파일 경로와 라벨(인덱스) 추가
        for cls_name in classes:
            cls_folder = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for file_name in os.listdir(cls_folder):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(cls_folder, file_name)
                    self.data.append((file_path, self.class_to_idx[cls_name]))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_path, label = self.data[idx]
        # print("wav_path: ", wav_path)

        # 오디오 로드
        waveform, sample_rate = torchaudio.load(wav_path)  # waveform shape: [channels, time]

        # 모노로 변환
        if waveform.shape[0] == 2:  # 2채널(스테레오)라면 모노로 변환
            waveform = torch.mean(waveform, dim=0, keepdim=True)  #torch.Size([2, 88200]) -> torch.Size([1, 88200])

        combined_image = create_combined_plots(wav_path)
        # RGBA 이미지를 그레이스케일로 변환 (모든 채널의 평균)
        grayscale = combined_image.mean(dim=2)  # [높이, 너비]
        combined_features = grayscale.unsqueeze(0).unsqueeze(0)  # (4, 높이, 너비) -> (1, 1, 높이, 너비)
    
        return combined_features, label