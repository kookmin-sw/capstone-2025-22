# -- type
TYPE_QUARTER = "quarter"
TYPE_EIGHTH = "eighth"
TYPE_HALF = "half"
TYPE_WHOLE = "whole"
TYPE_16th = "16th"

# -- duration
DURATION_0250 = "0.250"
DURATION_0375 = "0.375"
DURATION_0500 = "0.500"
DURATION_0750 = "0.750"
DURATION_1000 = "1.000"
DURATION_1500 = "1.500"
DURATION_2000 = "2.000"
DURATION_3000 = "3.000"
DURATION_4000 = "4.000"

# -- rest
REST_QUARTER = "REST_QUARTER"
REST_EIGHTH = "REST_EIGHTH"
REST_HALF = "REST_HALF"
REST_WHOLE = "REST_WHOLE"
REST_16th = "REST_16th"

REST_NOTES = [
    REST_QUARTER,
    REST_EIGHTH,
    REST_HALF,
    REST_WHOLE,
    REST_16th,
]

REST2DURATION = {
    REST_QUARTER: DURATION_1000,
    REST_EIGHTH: DURATION_0500,
    REST_HALF: DURATION_2000,
    REST_WHOLE: DURATION_4000,
    REST_16th: DURATION_0250,
}


# -- pitchs
PITCHS = [
    "D4",
    "F4",
    "A4",
    "C5",
    "D5",
    "E5",
    "F5",
    "G5",
    "A5",
    "B5",
]
PITCH_NOTES = PITCHS + REST_NOTES
PTICH_HEIGHT = len(PITCH_NOTES)
# {0: 'A3', 1: 'B3', 2: 'C4', 3: 'D4', 4: 'E4', 5: 'F4', 6: 'G4', 7: 'A4', 8: 'B4', 9: 'C5', 10: 'D5', 11: 'E5', 12: 'F5', 13: 'G5', 14: 'A5', 15: 'B5', 16: 'C6'}
CODE2PITCH_NOTE = {index: note for index, note in enumerate(PITCH_NOTES)}
# {'A3': 0, 'B3': 1, 'C4': 2, 'D4': 3, 'E4': 4, 'F4': 5, 'G4': 6, 'A4': 7, 'B4': 8, 'C5': 9, 'D5': 10, 'E5': 11, 'F5': 12, 'G5': 13, 'A5': 14, 'B5': 15, 'C6': 16}
PITCH_NOTE2CODE = {note: index for index, note in enumerate(PITCH_NOTES)}

# -- duration
# 4분 음표 기준 1
DURATIONS = [
    DURATION_0250,
    DURATION_0375,
    DURATION_0500,
    DURATION_0750,
    DURATION_1000,
    DURATION_1500,
    DURATION_2000,
    DURATION_3000,
    DURATION_4000,
]
DURATIONS2TYPE = {
    DURATION_0250: TYPE_16th,
    DURATION_0375: TYPE_16th,  # -- (임시)
    DURATION_0500: TYPE_EIGHTH,
    DURATION_0750: TYPE_EIGHTH,  # -- (임시)
    DURATION_1000: TYPE_QUARTER,
    DURATION_1500: TYPE_QUARTER,  # -- (임시)
    DURATION_2000: TYPE_HALF,
    DURATION_3000: TYPE_HALF,  # -- (임시)
    DURATION_4000: TYPE_WHOLE,
}
DURATION_NOTES = DURATIONS + REST_NOTES
DURATION_HEIGHT = len(DURATION_NOTES)
# {0: 0.25, 1: 0.375, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.5, 6: 2.0, 7: 3.0, 8: 4.0, 9: 'REST_QUARTER', 10: 'REST_EIGHTH', 11: 'REST_HALF', 12: 'REST_WHOLE'}
CODE2DURATION_NOTE = {index: note for index, note in enumerate(DURATION_NOTES)}
# {0.25: 0, 0.375: 1, 0.5: 2, 0.75: 3, 1.0: 4, 1.5: 5, 2.0: 6, 3.0: 7, 4.0: 8, 'REST_QUARTER': 9, 'REST_EIGHTH': 10, 'REST_HALF': 11, 'REST_WHOLE': 12}
DURATION_NOTE2CODE = {note: index for index, note in enumerate(DURATION_NOTES)}

# -- note (pitch + duration)
NOTES = PITCHS + DURATIONS + REST_NOTES
NOTES_HEIGHT = len(NOTES)
# {0: 'D4', 1: 'F4', 2: 'A4', 3: 'C5', 4: 'D5', 5: 'E5', 6: 'F5', 7: 'G5', 8: 'A5', 9: 'B5', 10: 0.25, 11: 0.375, 12: 0.5, 13: 0.75, 14: 1.0, 15: 1.5, 16: 2.0, 17: 3.0, 18: 4.0, 19: 'REST_QUARTER', 20: 'REST_EIGHTH', 21: 'REST_HALF', 22: 'REST_WHOLE', 23: 'REST_16th'}
CODE2NOTES = {index: note for index, note in enumerate(NOTES)}
NOTES2CODE = {note: index for index, note in enumerate(NOTES)}
