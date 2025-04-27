import cv2
import numpy as np
from music21 import *

# 이미지 불러오기
img_path = r"D:\PSY\Semester0501\Capstone_Project\letmedrum\drum_sheet_music.png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image at path '{img_path}' could not be loaded.")

src = img.copy()

# import cv2
# import numpy as np

# def detect_measure_lines(gray):
#     height, width = gray.shape
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 10))
#     detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
#     contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     measure_lines = [cv2.boundingRect(cnt)[0] for cnt in contours]
#     measure_lines.sort()
#     return measure_lines

# # Load image
# gray = cv2.imread("score.png", cv2.IMREAD_GRAYSCALE)
# if gray is None:
#     raise FileNotFoundError("Processed score image not found.")

# # Detect measure lines
# measure_lines = detect_measure_lines(gray)

# # Split into groups of 4 measures
# num_measures = len(measure_lines)
# measures_per_part = 4
# split_images = []

# for i in range(0, num_measures, measures_per_part):
#     if i + measures_per_part < num_measures:
#         x_start = measure_lines[i]
#         x_end = measure_lines[i + measures_per_part] if i + measures_per_part < num_measures else gray.shape[1]
#         split_img = gray[:, x_start:x_end]
#         split_images.append(split_img)
#         cv2.imwrite(f'measure_part_{i//4+1}.png', split_img)

# print(f"Split into {len(split_images)} parts.")

# 이미지 전처리
# 이진화 및 변수 지정
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
height, width = gray.shape

# 보표 추출
mask = np.zeros(gray.shape, np.uint8)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)

for i in range(1, cnt):
    x, y, w, h, area = stats[i]
    if w > width * 0.5:
        roi = src[y:y+h, x:x+w]
        cv2.imwrite('line%s.png' %i, roi)
for i in range(1, cnt):
    x, y, w, h, area = stats[i]
    if w > width * 0.5:
        cv2.rectangle(mask, (x, y, w, h), (255, 255, 255), -1)
masked = cv2.bitwise_and(gray, mask)

# 오선 삭제
staves = []
for row in range(height):
    pixels = 0
    for col in range(width):
        pixels += (masked[row][col] == 255)
    if pixels >= width * 0.5:
        if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:
            staves.append([row, 0])
        else:
            staves[-1][1] += 1

for staff in range(len(staves)):
    top_pixel = staves[staff][0]
    bot_pixel = staves[staff][0] + staves[staff][1]
    for col in range(width):
        if height-staves[staff][1] > bot_pixel and masked[top_pixel - 1][col] == 0 and masked[bot_pixel + 1][col] == 0:
            for row in range(top_pixel, bot_pixel + 1):
                masked[row][col] = 0
cv2.imwrite('score.png', 255-masked)

# 객체 탐색
# 윤곽 검출
contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i=1
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = 255-masked[y-5:y+h+5, x-5:x+w+5]
    cv2.imwrite('save%s.jpg' %i, roi)
    i+=1
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(src, (x, y, w, h), (255, 0, 0), 2)

# 결과 저장
cv2.imwrite('result.png', src)

cv2.imshow('Result', src)
cv2.waitKey(0)
cv2.destroyAllWindows()