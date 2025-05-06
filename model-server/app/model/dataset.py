import os
import sys
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

import utils

class DrumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir에서 마지막 디렉토리 이름만 추출 (예: 'train')
        self.dataset_name = os.path.basename(os.path.normpath(data_dir))
        self.MAX_FRAMES = 86 #약 1초 고정크기
        self.pool = nn.AdaptiveAvgPool2d((128, self.MAX_FRAMES))

        # 하위 디렉토리를 포함한 모든 .wav 파일 찾기
        self.file_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.file_paths.append(os.path.join(root, file))

        self.transform = transform
        self.class_labels = ["KD", "SD", "CY", "TT", "HH"]

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        labels = self._get_labels_from_filename(wav_path) #ex. KD:SD_001.wav -> [1,1,0,0,0]

        features = utils.wav_to_mel(wav_path, self.transform) #ex. [1, 128=n_mels, T]

        _, _, T = features.shape
        if T < self.MAX_FRAMES:
            pad_amount = self.MAX_FRAMES - T
            features = F.pad(features, (0, pad_amount))

        features = self.pool(features)  # T가 달라도 고정크기로 변환

        return features, labels
    
    def _get_labels_from_filename(self, file_path):
        """파일명에서 레이블 정보를 추출하여 원-핫 인코딩 형태로 반환"""
        filename = os.path.basename(file_path)
        basename = os.path.splitext(filename)[0]  # 확장자 제거

        # 파일명에서 레이블 부분 추출 (예: "HH:CY_001" -> ["HH", "CY"])
        if '_' in basename:
            labels_part = basename.split('_')[0]
        else: 
            labels_part = basename
            
        label_names = labels_part.split(':')  # 콜론(:)으로 구분된 다중 레이블 처리

        # 원-핫 인코딩 형태로 레이블 생성 (멀티 레이블)
        labels = torch.zeros(len(self.class_labels))
        for label in label_names:
            if label in self.class_labels:
                label_idx = self.class_labels.index(label)
                labels[label_idx] = 1.0
                
        return labels
    
    def print_label_distribution(self):
        """데이터셋의 레이블 분포와 조합을 분석하여 출력"""
        # 단일 레이블 목록
        single_labels = ["KD", "SD", "CY", "TT", "HH"]
        
        # 다중 레이블 목록 (지정된 순서)
        multi_labels = ["KD:SD", "KD:CY", "KD:TT", "KD:HH", 
                         "SD:CY", "SD:TT", "SD:HH", 
                         "CY:TT", "CY:HH", 
                         "TT:HH"]
        
        # 모든 레이블 조합의 카운트를 저장할 딕셔너리 (기본값 0으로 초기화)
        label_counts = {label: 0 for label in single_labels}
        combo_counts = {combo: 0 for combo in multi_labels}
        
        # 단일/다중 레이블 파일 카운트
        single_label_count = 0
        multi_label_count = 0
        
        # 모든 파일 순회
        for file_path in self.file_paths:
            # 레이블 추출
            labels = self._get_labels_from_filename(file_path)
            
            # 활성화된 레이블 개수
            active_count = int(torch.sum(labels).item())
            
            # 활성화된 레이블 이름 목록
            active_labels = [self.class_labels[i] for i, is_active in enumerate(labels) if is_active]
            
            if active_count == 1:  # 단일 레이블
                single_label_count += 1
                # 레이블 카운트 증가
                label_counts[active_labels[0]] += 1
            elif active_count > 1:  # 다중 레이블
                multi_label_count += 1
                # 정렬된 레이블 조합 생성
                sorted_labels = sorted(active_labels, key=lambda x: single_labels.index(x))
                combo = ":".join(sorted_labels)
                # 유효한 다중 레이블 조합인 경우 카운트 증가
                if combo in combo_counts:
                    combo_counts[combo] += 1
        
        # 결과 출력
        total_files = len(self.file_paths)
        print(f"\n=== {self.dataset_name} 레이블 분포 통계 ===")
        print(f"총 wav 파일 수: {total_files}")
        print(f"단일 레이블 파일 수: {single_label_count} ({single_label_count/total_files*100:.1f}%)")
        print(f"다중 레이블 파일 수: {multi_label_count} ({multi_label_count/total_files*100:.1f}%)")
        
        print("\n--- 단일 레이블 분포 ---")
        for label in single_labels:
            count = label_counts[label]
            percentage = (count/total_files*100) if total_files > 0 else 0
            print(f"{label}: {count}개 ({percentage:.1f}%)")
        
        print("\n--- 다중 레이블 분포 ---")
        for combo in multi_labels:
            count = combo_counts[combo]
            percentage = (count/total_files*100) if total_files > 0 else 0
            print(f"{combo}: {count}개 ({percentage:.1f}%)")
        print("=============================\n")