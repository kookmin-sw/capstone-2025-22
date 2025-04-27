from constant.note import NOTES


STAVE_HEIGHT = 120
STAVE_WIDTH = 1000

# -- note좌우 pad 크기 [ex. note width 20 중에, 8 pad * 2 -> 4px]
NOTE_PAD = 2

# -- chunk length - model에 넣기 전 dataset 가공 시
CHUNK_TIME_LENGTH = 50

PREDICT_STD = 0.5

# -- extension
PNG = "PNG"
XML = "XML"
JSON = "JSON"
CSV = "CSV"
TXT = "txt"
EXP = {PNG: "png", XML: "xml", JSON: "json", CSV: "csv", TXT: "txt"}

STAVE = "stave"
PAD_STAVE = "pad-stave"
FEATURE = "feature"
LABELED_FEATURE = "labeled-feature"
CURSOR = "cursor"

OMR = "omr-seq2seq"

PITCH = "pitch"
NOTE = "note"
MULTI_CLASS = "multi-class"
MULTI_LABEL = "multi-label"
TRANSFORMER = "transformer"

AUGMENT = "augment"
ANNOTATION = "annotation"

"""
json info
"""
KEY_CURSOR_LIST = "cursorList"
KEY_MEASURE_LIST = "measureList"

ORIGIN = "origin"
AWGN = "awgn"
ET_SMALL = "et_small"
ET_LARGE = "et_large"
ALL_AUGMENT = "all_augment"

AUGMENT_SET = {
    ORIGIN: ORIGIN,
    # AWGN: AWGN,
    # ET_SMALL: ET_SMALL,
    # ET_LARGE: ET_LARGE,
    # ALL_AUGMENT: ALL_AUGMENT,
}


"""
-- dataframe 헤더 초기화
-- [G5, A5, ..., REST,...]
"""


def get_feature_cols(name):
    return [f"{STAVE}-{name}-{i + 1}" for i in range(STAVE_HEIGHT)]


DATASET_DF = {
    "label": NOTES,
    ORIGIN: get_feature_cols(ORIGIN),
    AWGN: get_feature_cols(AWGN),
    ET_SMALL: get_feature_cols(ET_SMALL),
    ET_LARGE: get_feature_cols(ET_LARGE),
    ALL_AUGMENT: get_feature_cols(ALL_AUGMENT),
}
