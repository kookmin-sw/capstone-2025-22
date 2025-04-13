import os
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import numpy as np

class DrumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 하위 디렉토리를 포함한 모든 .wav 파일 찾기
        self.file_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.file_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.file_paths)} .wav files in {data_dir} and its subdirectories")
        
        self.transform = transform
        self.class_labels = ["KD", "SD", "CY", "TT", "HH"]

        # 레이블 분포 출력
        self.print_label_distribution()

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        # print("wav_path: ", wav_path)

        # 파일 확장자 제외한 파일명 추출
        filename = os.path.basename(wav_path)
        basename = os.path.splitext(filename)[0]  # 확장자 제거

        # 파일명에서 레이블 정보 추출 (예: "HH:CY_001" -> ["HH", "CY"])
        if '_' in basename:
            labels_part = basename.split('_')[0]
        else: labels_part = basename
            
        label_names = labels_part.split(':') # 콜론(:)으로 구분된 다중 레이블 처리

        # 원-핫 인코딩 형태로 레이블 생성 (멀티 레이블)
        labels = torch.zeros(len(self.class_labels))
        for label in label_names:
            if label in self.class_labels:
                label_idx = self.class_labels.index(label)
                labels[label_idx] = 1.0

        # 오디오 로드
        waveform, sample_rate = torchaudio.load(wav_path)  # waveform shape: [channels, time]
        if waveform.shape[0] > 1:  # 2채널(스테레오)라면 모노로 변환
            waveform = torch.mean(waveform, dim=0, keepdim=True)  #torch.Size([2, 88200]) -> torch.Size([1, 88200])

        # 오디오 -> 멜 스펙토그램
        features = self.transform(waveform)
        # 데이터 전처리 (정규화 등)
        if len(features.shape) != 3:  # [높이, 너비]인 경우
            features = features.unsqueeze(0)  # [채널=1, 높이, 너비]로 채널 차원 추가 (배치 처리는 DataLoader 담당)

        # AdaptiveAvgPool2d를 사용하여 고정된 크기로 변환
        adaptive_pool = torch.nn.AdaptiveAvgPool2d((16, 16))  # 높이와 너비를 16로 고정
        features = adaptive_pool(features)

        return features, labels
    
    def print_label_distribution(self):
        """
        데이터셋의 각 레이블별 개수를 출력하는 메서드
        """
        # 각 레이블별 카운터 초기화
        label_counts = {label: 0 for label in self.class_labels}
        # 단일 레이블 파일 수
        single_label_files = 0
        # 다중 레이블 파일 수
        multi_label_files = 0
        # 레이블이 없는 파일 수
        no_label_files = 0
        
        # 모든 파일 순회
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            basename = os.path.splitext(filename)[0]
            
            if '_' in basename:
                labels_part = basename.split('_')[0]
            else:
                labels_part = basename
                
            # 콜론으로 구분된 레이블 처리
            label_names = labels_part.split(':')
            
            # 유효한 레이블만 필터링
            valid_labels = [label for label in label_names if label in self.class_labels]
            
            if len(valid_labels) == 0:
                no_label_files += 1
            elif len(valid_labels) == 1:
                single_label_files += 1
            else:
                multi_label_files += 1
            
            # 각 레이블 카운트 증가
            for label in valid_labels:
                if label in self.class_labels:
                    label_counts[label] += 1
        
        # 결과 출력
        print("\n===== 데이터셋 레이블 분포 =====")
        print(f"총 파일 수: {len(self.file_paths)}")
        print(f"단일 레이블 파일: {single_label_files}")
        print(f"다중 레이블 파일: {multi_label_files}")
        print(f"레이블 없는 파일: {no_label_files}")
        print("\n각 레이블별 분포:")
        
        # 각 레이블별 개수와 비율 출력
        for label, count in label_counts.items():
            percentage = (count / len(self.file_paths)) * 100
            print(f"  - {label}: {count}개 ({percentage:.1f}%)")
            
        print("=============================\n")