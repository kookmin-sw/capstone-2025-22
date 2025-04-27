import cv2
from constant.common import AUGMENT, EXP, MULTI_LABEL, NOTE, PNG, XML
from constant.path import DATA_TEST_PATH, IMAGE_PATH, RESULT_XML_PATH
from model.multilabel_note_model import MultiLabelNoteModel
from feature_labeling import FeatureLabeling
from data_processing import DataProcessing
from Image2augment import Image2Augment
from score2stave import Score2Stave
from note2xml import Note2XML
from util import Util
from xml2annotation import Xml2Annotation


# ======================== omr ai ===========================
def process_all_data_aug():
    Image2Augment.process_all_image2augment()


def process_all_data():
    # Xml2Annotation.process_xml2annotation(
    #     "../data/raw/osmd-dataset-v1.0.0/Rock-ver/Rock-ver.xml"
    # )
    # # save feature csv
    DataProcessing.process_all_score2stave()

    # # save labeled-feature csv
    # FeatureLabeling.process_all_feature2label()


# multilabel_pitch_model = MultiLabelPitchModel(40, 0.001, 32, MULTI_LABEL, PITCH)
# multilabel_pitch_model = MultiLabelNoteModel(40, 0.001, 32, MULTI_LABEL, NOTE)


# def train_model():
#     # get feature, label from csv, train
#     multilabel_pitch_model.create_dataset()
#     multilabel_pitch_model.create_model()

#     multilabel_pitch_model.train()
#     multilabel_pitch_model.evaluate()
#     multilabel_pitch_model.save_model()


# def predict_model():
#     multilabel_pitch_model.load_model()
#     predict_test_datas = [
#         # "didyouloveme_pad-stave_0_2024-04-06_16-26-04.png",
#         # "Rock-ver_pad-stave_9_2024-04-06_17-22-00.png",
#         # "uptownfunk_pad-stave_17_2024-04-06_16-26-10.png",
#         "Rock-ver_pad-stave_11_2024-04-06_17-22-00.png",
#     ]
#     for predict_test_data in predict_test_datas:
#         note_data = multilabel_pitch_model.predict_score(
#             f"{DATA_TEST_PATH}/{predict_test_data}"
#         )
#         datetime = Util.get_datetime()
#         Note2XML.create_musicxml(
#             note_data, f"{RESULT_XML_PATH}/musicxml-{datetime}.{EXP[XML]}"
#         )
#     # files = Util.get_all_files(f"{DATA_TEST_PATH}", EXP[PNG])
#     # for file_path in files:
#     #     title = Util.get_title_from_dir(file_path)
#     #     note_data = multilabel_pitch_model.predict_score(f"{file_path}")
#     #     datetime = Util.get_datetime()
#     #     Note2XML.create_musicxml(
#     #         note_data, f"{RESULT_XML_PATH}/musicxml-{title}-{datetime}.{EXP[XML]}"
#     #     )


# process_all_data_aug()
process_all_data()
# train_model()
# predict_model()
