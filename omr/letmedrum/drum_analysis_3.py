import music21
from tensorflow.keras import layers

# 1. 악보 로드
path = 'output.xml'
score = music21.converter.parse(path)

# 2. 기호+박자 분석용 배열
score_arr = []

for element in score.recurse():
    if isinstance(element, music21.note.Unpitched):
        note_name = element.displayName  # 타악기 이름 (Snare Drum, Hi-Hat 등)
        duration = element.duration.quarterLength
        score_arr.append([note_name, duration])

# 3. 시간 계산 (선택사항 - 필요 없으면 제거 가능)
second = ['0', ]
for i in range(len(score_arr)):
    second.append(str(float(second[i]) + score_arr[i][1] * 0.5))

# 4. 매핑 함수
def replace(L):
    duration_mapping = {
        4.0: '4분의 4음표',
        3.0: '4분의 3음표',
        2.0: '2분의 1음표',
        1.5: '8분의 3음표',
        1.0: '4분의 1음표',
        0.5: '8분의 1음표',
        0.25: '16분의 1음표'
    }

    for i, sublist in enumerate(L):
        if isinstance(sublist[1], float):
            L[i][1] = duration_mapping.get(sublist[1], sublist[1])

    return L

replaced_score = replace(score_arr)

char_to_int_mapping = [
    "|",
    "barline",
    "clef-percussion",
    "timeSignature-4/4",
    "note-F4_eighth",
    "note-F4_eighth.",
    "note-F4_half",
    "note-F4_half.",
    "note-F4_quarter",
    "note-F4_quarter.",
    "note-F4_16th",
    "note-F4_16th.",
    "note-F4_whole",
    "note-F4_whole.",
    "note-F4_32nd",
    "note-C5_eighth",
    "note-C5_eighth.",
    "note-C5_half",
    "note-C5_half.",
    "note-C5_quarter",
    "note-C5_quarter.",
    "note-C5_16th",
    "note-C5_16th.",
    "note-C5_whole",
    "note-C5_whole.",
    "note-C5_32nd",
    "note-G5_eighth",
    "note-G5_eighth.",
    "note-G5_half",
    "note-G5_half.",
    "note-G5_quarter",
    "note-G5_quarter.",
    "note-G5_16th",
    "note-G5_16th.",
    "note-G5_whole",
    "note-G5_whole.",
    "note-G5_32nd",
]
char_to_num = layers.StringLookup(vocabulary=list(char_to_int_mapping), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# 5. 텍스트 출력
sentence = []
for l in replaced_score:
    sentence.append(f"{l[1]} {l[0]}입니다.\n")

print(''.join(sentence))

# 6. 텍스트 파일로 저장
with open('drum_mxl2txt.txt', 'w', encoding='utf-8') as f:
    f.writelines(sentence)

print("✅ 드럼 악보 분석 완료 → 'drum_mxl2txt.txt' 저장 완료!")