import os
import re
import sys

from rename import extract_prefix_and_number

previous_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(previous_folder)
from src.CNN.multiCNN.dataset import DrumDataset

def normalize_filenames(directory):
    """
    디렉토리 내의 파일 이름을 정규화합니다.
    - 다중 레이블 순서를 통일 (예: SD:KD → KD:SD)
    - 번호 순차적으로 재지정
    """
    # 유효한 단일 레이블
    valid_single_labels = ["KD", "SD", "CY", "TT", "HH"]
    
    # 유효한 다중 레이블 조합 (순서 통일됨)
    valid_multi_labels = [
        "KD:SD", "KD:CY", "KD:TT", "KD:HH", 
        "SD:CY", "SD:TT", "SD:HH", 
        "CY:TT", "CY:HH", 
        "TT:HH"
    ]
    
    # 레이블 카운터 초기화
    label_counters = {label: 1 for label in valid_single_labels + valid_multi_labels}
    
    # 파일 목록 가져오기
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    # 변경 사항 기록
    renamed_count = 0
    no_change_count = 0
    skipped_count = 0
    
    for filename in wav_files:
        # 파일명에서 레이블과 번호 추출
        prefix, number = extract_prefix_and_number(filename)
        
        if not prefix or not number:
            skipped_count += 1
            print(f"건너뜀: {filename} (형식 불일치)")
            continue
        
        # 정규화된 레이블 생성
        normalized_prefix = normalize_label(prefix, valid_single_labels)
        
        if normalized_prefix != prefix:
            # 레이블이 변경된 경우
            old_path = os.path.join(directory, filename)
            new_filename = f"{normalized_prefix}_{label_counters[normalized_prefix]}.wav"
            new_path = os.path.join(directory, new_filename)
            
            # 파일 이름 변경
            os.rename(old_path, new_path)
            label_counters[normalized_prefix] += 1
            renamed_count += 1
            print(f"변경됨: {filename} → {new_filename}")
        else:
            # 레이블이 이미 정규화된 경우
            no_change_count += 1
    
    print(f"\n완료: {renamed_count}개 파일 이름 변경됨, {no_change_count}개 변경 없음, {skipped_count}개 건너뜀")

def normalize_label(label, valid_single_labels):
    """
    레이블을 정규화합니다:
    1. 단일 레이블이면 그대로 반환
    2. 다중 레이블이면 정렬하여 표준 형식으로 변환
    """
    if ":" not in label:
        # 단일 레이블은 그대로 반환
        return label
    
    # 다중 레이블인 경우 레이블 분리
    parts = label.split(":")
    
    # 유효하지 않은 레이블이 있는지 확인
    for part in parts:
        if part not in valid_single_labels:
            return label  # 유효하지 않은 레이블이 있으면 원본 반환
    
    # 레이블을 정렬하여 통일된 순서로 변환
    sorted_parts = sorted(parts, key=lambda x: valid_single_labels.index(x))
    
    # 정렬된 레이블을 다시 합침
    return ":".join(sorted_parts)

def reindex_filenames(directory):
    """
    디렉토리 내의 파일 이름의 번호를 순차적으로 재할당합니다.
    """
    # 유효한 단일 레이블
    valid_single_labels = ["KD", "SD", "CY", "TT", "HH"]
    
    # 유효한 다중 레이블 조합
    valid_multi_labels = [
        "KD:SD", "KD:CY", "KD:TT", "KD:HH", 
        "SD:CY", "SD:TT", "SD:HH", 
        "CY:TT", "CY:HH", 
        "TT:HH"
    ]
    
    # 모든 레이블 유형
    all_labels = valid_single_labels + valid_multi_labels
    
    # 레이블별 파일 목록
    label_files = {label: [] for label in all_labels}
    
    # 파일 목록 가져오기 및 레이블별로 분류
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            prefix, _ = extract_prefix_and_number(filename)
            if prefix in all_labels:
                label_files[prefix].append(filename)
    
    # 변경 사항 기록
    renamed_count = 0
    
    # 각 레이블에 대해 번호 재할당
    for label, files in label_files.items():
        if not files:
            continue
            
        # 파일명의 번호순으로 정렬
        sorted_files = sorted(files, key=lambda f: int(extract_prefix_and_number(f)[1]))
        
        # 번호 재할당
        for i, filename in enumerate(sorted_files, 1):
            new_filename = f"{label}_{i}.wav"
            
            # 이미 올바른 번호를 가진 경우 건너뜀
            if filename == new_filename:
                continue
                
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 임시 파일명으로 먼저 변경 (충돌 방지)
            temp_path = os.path.join(directory, f"temp_{i}_{label}.wav")
            os.rename(old_path, temp_path)
            
            renamed_count += 1
            print(f"번호 재할당: {filename} → {new_filename}")
    
    # 임시 파일명을 최종 파일명으로 변경
    for filename in os.listdir(directory):
        if filename.startswith("temp_") and filename.endswith(".wav"):
            match = re.match(r"temp_(\d+)_(.+)\.wav", filename)
            if match:
                index = match.group(1)
                label = match.group(2)
                new_filename = f"{label}_{index}.wav"
                
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                os.rename(old_path, new_path)
    
    print(f"\n완료: {renamed_count}개 파일 번호 재할당됨")

# 전체 정규화 프로세스 실행 함수
def standardize_drum_files(directory):
    """
    드럼 파일 이름을 정규화하고 번호를 재할당합니다.
    """
    print(f"\n=== 파일명 정규화 시작: {directory} ===")
    # 1단계: 레이블 정규화
    normalize_filenames(directory)
    # 2단계: 번호 순차적 재할당
    reindex_filenames(directory)
    print("=== 파일명 정규화 완료 ===\n")

# 사용 예시:
if __name__ == "__main__":
    train_dir = "/Users/ddps/Desktop/2025-1/Capstone/letmedrum/data/train"
    val_dir = "/Users/ddps/Desktop/2025-1/Capstone/letmedrum/data/val"
    
    # standardize_drum_files(train_dir)
    # standardize_drum_files(val_dir)

    train_dataset = DrumDataset('train')
    val_dataset   = DrumDataset('val')
