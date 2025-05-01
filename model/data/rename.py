import os
import shutil
import re

def rename_drum_files(directory):
    # 드럼 타입 매핑
    drum_type_mapping = {
        'toms': 'TT',
        'snare': 'SD',
        'overheads': 'CY',
        'kick': 'KD'
    }
    
    # 디렉토리 내의 모든 폴더 순회
    for root, dirs, files in os.walk(directory):
        # 현재 폴더명 가져오기
        current_folder = os.path.basename(root).lower()
        
        # 해당 폴더가 매핑에 있는 경우에만 처리
        if current_folder in drum_type_mapping:
            prefix = drum_type_mapping[current_folder]
            
            # 폴더 내의 모든 파일 처리
            for file in files:
                if file.endswith('.wav'):
                    # 파일 번호 추출 (예: "Tom Sample 1.wav" -> "1")
                    try:
                        number = ''.join(filter(str.isdigit, file))
                        if not number:
                            number = '1'  # 숫자가 없는 경우 기본값
                        
                        # 새 파일명 생성
                        new_filename = f"{prefix}_{number}.wav"
                        
                        # 전체 경로 생성
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, new_filename)
                        
                        # 파일 이름 변경
                        os.rename(old_path, new_path)
                        print(f"Renamed: {file} -> {new_filename}")
                    except Exception as e:
                        print(f"Error renaming {file}: {str(e)}")

def extract_prefix_and_number(filename):
    """파일명에서 접두어와 숫자를 추출합니다."""
    # 언더바로 분리하여 접두어와 숫자 부분으로 나눔
    parts = os.path.splitext(filename)[0].split('_')
    
    if len(parts) != 2:
        return None, None
    
    prefix = parts[0]  # 접두어 부분 (예: "KD" 또는 "KD:HH")
    
    # 숫자 부분 추출 및 변환
    try:
        number = int(parts[1])
        return prefix, number
    except ValueError:
        return None, None

def move_files_with_renumbering(source_dir, target_dir, prefix=None, limit=None):
    # 대상 디렉토리 존재 확인 및 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"대상 디렉토리 생성됨: {target_dir}")
    
    if prefix:
        print(f"접두어 '{prefix}'로 시작하는 파일만 처리합니다")
    
    # 대상 디렉토리의 기존 파일 분석하여 최대 숫자 찾기
    prefix_max_numbers = {}
    
    for filename in os.listdir(target_dir):
        if filename.endswith('.wav'):
            # 파일명에서 접두어와 숫자 추출 (예: CY_123.wav 또는 CY:HH_123.wav)
            file_prefix, number = extract_prefix_and_number(filename)
            if file_prefix and number:
                # 지정된 접두어가 없거나 정확히 일치하는 접두어인 경우만 처리
                if not prefix or file_prefix == prefix:
                    if file_prefix in prefix_max_numbers:
                        prefix_max_numbers[file_prefix] = max(prefix_max_numbers[file_prefix], number)
                    else:
                        prefix_max_numbers[file_prefix] = number
    
    # 이동한 파일 카운터
    moved_count = 0
    renamed_count = 0
    skipped_count = 0
    
    # 소스 디렉토리의 파일 이동
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            # 파일명에서 접두어와 숫자 추출
            file_prefix, number = extract_prefix_and_number(filename)
            
            if not file_prefix or not number:
                continue  # 형식이 맞지 않는 파일은 건너뜀
            
            # 접두어가 정확히 일치하는지 확인
            if prefix and file_prefix != prefix:
                skipped_count += 1
                continue  # 지정된 접두어와 정확히 일치하지 않으면 건너뜀
            
            # 새 파일명 결정
            new_number = number
            
            if file_prefix in prefix_max_numbers:
                # 대상 디렉토리에 같은 접두어 파일이 있으면 숫자 조정
                new_number = number + prefix_max_numbers[file_prefix]
            
            source_path = os.path.join(source_dir, filename)
            
            # 원본 파일명이 그대로 유지되는 경우
            if new_number == number:
                new_filename = filename
            else:
                # 번호가 변경되는 경우, 접두어는 유지하면서 숫자만 변경
                new_filename = file_prefix + f"_{new_number}.wav"
                renamed_count += 1
            
            target_path = os.path.join(target_dir, new_filename)
            
            # 파일이 이미 존재하는지 확인
            if os.path.exists(target_path):
                # 접두어의 최대 번호 + 1 사용
                if file_prefix not in prefix_max_numbers:
                    prefix_max_numbers[file_prefix] = 0
                prefix_max_numbers[file_prefix] += 1
                new_number = prefix_max_numbers[file_prefix]
                new_filename = file_prefix + f"_{new_number}.wav"
                target_path = os.path.join(target_dir, new_filename)
                renamed_count += 1
            
            # 파일 이동 (복사 대신 이동으로 변경)
            shutil.move(source_path, target_path)
            moved_count += 1
            print(f"파일 이동: {filename} -> {new_filename}")

            if limit and moved_count >= limit: break
    
    print(f"\n완료: {moved_count}개 파일 이동됨 (그 중 {renamed_count}개 파일명 변경됨, {skipped_count}개 건너뜀)")



# 스크립트 실행
if __name__ == "__main__":
    # train과 val 폴더 모두 처리
    data_dir = "/Users/ddps/Desktop/2025-1/Capstone/letmedrum/data"
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # print("Processing training data...")
    # rename_drum_files(train_dir)
    
    # print("\nProcessing validation data...")
    # rename_drum_files(val_dir)
    # 실행
    # source_directory = "/Users/ddps/Desktop/2025-1/Capstone/letmedrum/etc/drum/cut"
    source_directory = "/Users/ddps/Desktop/2025-1/Capstone/letmedrum/etc/drum/padded_data"
    # source_directory = train_dir
    # target_directory, limit = train_dir, 578
    target_directory, limit = val_dir, 200
    prefix = "CY"

    #KD 240+200
    #SD 24+200
    #CY 174+200'
    #TT 578+200
    #HH 0+200

    move_files_with_renumbering(source_directory, target_directory, prefix, limit=limit)