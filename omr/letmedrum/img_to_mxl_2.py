# # import  aspose.cells 
# # from aspose.cells import Workbook
# # workbook = Workbook("result.png")
# # workbook.save("result.xml")

# import cv2
# import numpy as np
# import music21

# # 1. 이미지 불러오기
# img_path = "result.png"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# if img is None:
#     raise FileNotFoundError(f"Image at path '{img_path}' could not be loaded.")

# # 2. 이미지 이진화 (흑백 변환)
# _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# # 3. 윤곽선 검출 (드럼 기호 찾기)
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 4. 드럼 기호별 매핑 정의
# drum_mapping = {
#     "snare": "Snare Drum",
#     "hihat": "Closed Hi-Hat",
#     "bass": "Bass Drum"
# }

# # 5. MusicXML 악보 생성
# score = music21.stream.Score()
# part = music21.stream.Part()
# part.insert(0, music21.instrument.Percussion())  # 드럼 전용 악기 설정

# # 6. 기호별 음표 변환
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)

#     # 기호 크기 기반으로 드럼 악기 판별 (예제 기준)
#     if h > 50:  # 큰 기호 = 베이스 드럼
#         drum_type = "bass"
#     elif 30 < h <= 50:  # 중간 크기 = 스네어
#         drum_type = "snare"
#     else:  # 작은 기호 = 하이햇
#         drum_type = "hihat"

#     # 박자 랜덤 할당 (기본적으로 8분음표로 설정)
#     duration = 0.5  

#     # MusicXML용 드럼 노트 생성
#     note = music21.note.Unpitched()
#     note.displayName = drum_mapping[drum_type]  # 드럼 기호 적용
#     note.duration = music21.duration.Duration(duration)

#     # 악보에 추가
#     part.append(note)

# # 7. 악보를 Score에 추가 후 MusicXML 저장
# score.append(part)
# musicxml_path = "output.musicxml"
# score.write('musicxml', fp=musicxml_path)

# print(f"✅ 드럼 악보 변환 완료! → '{musicxml_path}' 저장됨 🎵")

import cv2
import numpy as np
import pytesseract
from music21 import stream, note

# 🎵 1. 이미지 불러오기 & 전처리
image = cv2.imread("result.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# 🎵 2. OCR로 드럼 악보 인식
text = pytesseract.image_to_string(thresh, config="--psm 6")
print("인식된 악보 기호:", text)

# 🎵 3. 기호 → MIDI 변환
drum_mapping = {"B": "B4", "S": "C5", "H": "G5"}  # 베이스, 스네어, 하이햇
detected_notes = text.split()
midi_notes = [drum_mapping[n] for n in detected_notes if n in drum_mapping]

# 🎵 4. MusicXML 생성
drum_score = stream.Score()
drum_part = stream.Part()

for midi_note in midi_notes:
    drum_note = note.Note(midi_note)
    drum_note.duration.quarterLength = 1
    drum_part.append(drum_note)

drum_score.append(drum_part)
drum_score.write("musicxml", fp="drum_output.musicxml")

print("🎼 MusicXML 파일이 생성되었습니다! → drum_output.musicxml")