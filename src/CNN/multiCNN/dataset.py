import os
import torch
import torchaudio
from torch.utils.data import Dataset

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
        if waveform.shape[0] == 2:  # 2채널(스테레오)라면 모노로 변환
            waveform = torch.mean(waveform, dim=0, keepdim=True)  #torch.Size([2, 88200]) -> torch.Size([1, 88200])

        # 오디오 -> 멜 스펙토그램으로 변환
        features = self.transform(waveform)

        return features, label