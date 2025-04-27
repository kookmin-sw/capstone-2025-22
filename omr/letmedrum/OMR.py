from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from music21 import stream, note, percussion

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) for img in images]

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def extract_symbols(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        symbol = pytesseract.image_to_string(roi, config='--psm 10')
        symbols.append((symbol.strip(), x))
    return sorted(symbols, key=lambda s: s[1])

def convert_to_musicxml(symbols):
    music = stream.Stream()
    for sym, _ in symbols:
        if sym == 'o':
            n = note.Note("C4", quarterLength=1)
            n.stemDirection = 'up'
        elif sym == 'x':
            n = note.Note("C5", quarterLength=1)
            n.stemDirection = 'up'
            n.notehead = 'x'
        else:
            continue
        music.append(n)
    return music

def drum_pdf_to_musicxml(pdf_path, output_path):
    images = pdf_to_images(pdf_path)
    music = stream.Score()
    
    for img in images:
        processed_img = preprocess_image(img)
        symbols = extract_symbols(processed_img)
        part = convert_to_musicxml(symbols)
        music.append(part)
    
    music.write('musicxml', fp=output_path)

# 실행 예제
drum_pdf_to_musicxml('letmedrum/drum_sheet_music_1.pdf', 'output.musicxml')