import os
import re
import glob
import argparse

from omegaconf import OmegaConf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
import xml.etree.ElementTree as ET
from datetime import datetime
from music21 import stream, instrument, clef, meter, note, percussion
from music21.musicxml import m21ToXml
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter, map_coordinates

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



def update_paths(base_path, paths):
    """
    Recursively update all paths in the dictionary by prepending the base_path.
    """
    if isinstance(paths, dict):
        for key, value in paths.items():
            paths[key] = update_paths(base_path, value)
    elif isinstance(paths, str):
        paths = os.path.join(base_path, paths)
    return paths


def getconfig(configpath):
    args = OmegaConf.load(configpath)

    workspace = os.path.dirname(configpath)
    args.filepaths = update_paths(workspace, args.filepaths)

    return args

class Util:
    @staticmethod
    def get_datetime():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    @staticmethod
    def get_title(score_path):
        return os.path.basename(os.path.dirname(score_path))
    @staticmethod
    def get_title_from_dir(score_path):
        return os.path.basename(score_path)
    @staticmethod
    def get_title_from_featurepath(score_path):
        return os.path.basename(os.path.dirname(os.path.dirname(score_path)))
    @staticmethod
    def get_all_files(parent_folder_path, exp):
        try:
            all_file_list = glob.glob(f"{parent_folder_path}/*")
            file_list = [file for file in all_file_list if file.endswith(f".{exp}")]
            return file_list
        except:
            print(f"!!-- {exp} 파일이 없습니다 : {parent_folder_path} --!!")
    @staticmethod
    def get_all_subfolders(folder_path):
        return [os.path.join(folder_path, item) for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
    @staticmethod
    def print_step(step):
        print(f"*************************************************")
        print(f"***************    {step}    ********************")
        print(f"*************************************************")
    @staticmethod
    def read_txt_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()
        content = [line.strip() for line in content]
        return content[0]
    @staticmethod
    def get_filepath_from_title(args, title, exp):
        path = f"{args.filepaths.raw_path.osmd}/{title}"
        try:
            file = Util.get_all_files(path, exp)
            return file[0]
        except:
            print(f"!!-- {exp} 파일이 없습니다 : {path} --!!")

class Image2Augment:
    @staticmethod
    def readimg(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        biImg = Image2Augment.img2binary(img)
        return biImg
    @staticmethod
    def img2binary(img):
        if img is None:
            raise RuntimeError("Image not found or unable to load.")
        if img.ndim == 3 and img.shape[-1] == 4:
            bgr = img[:, :, :3]
            alpha_channel = img[:, :, 3]
            white_background = np.all(bgr == [255, 255, 255], axis=-1)
            alpha_channel[white_background] = 0
            mask = 255 - alpha_channel
            gray = mask
        elif img.ndim == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise RuntimeError("Unsupported image type!")
        _, biImg = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return biImg
    @staticmethod
    def process_image2augment(args, rgb_path):
        biImg = Image2Augment.readimg(rgb_path)
        biImg = 255 - biImg
        awgn_image = Image2Augment.apply_awgn(biImg, args.augment.awgn_sigma)
        et_small_image = Image2Augment.apply_elastic_transform(biImg, args.augment.et_small_alpha_training, args.augment.et_small_sigma_evaluation)
        et_large_image = Image2Augment.apply_elastic_transform(biImg, args.augment.et_large_alpha_training, args.augment.et_large_sigma_evaluation)
        all_augmentations_image = Image2Augment.apply_all_augmentations(biImg, args.augment.et_small_alpha_training, args.augment.et_small_sigma_evaluation, args.augment.et_large_alpha_training, args.augment.et_large_sigma_evaluation)
        result = [("origin", biImg), ("awgn", awgn_image), ("et_small", et_small_image), ("et_large", et_large_image), ("all_augment", all_augmentations_image)]
        return result
    @staticmethod
    def apply_awgn(image, sigma=0.1):
        noisy_image = random_noise(image, mode="gaussian", var=sigma**2)
        return (255 * noisy_image).astype(np.uint8)
    @staticmethod
    def apply_elastic_transform(image, alpha, sigma):
        random_state = np.random.RandomState(None)
        height, width = image.shape[:2]
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        distorted_x = np.clip(x + dx, 0, width - 1)
        distorted_y = np.clip(y + dy, 0, height - 1)
        distorted_image = map_coordinates(image, [distorted_y, distorted_x], order=1, mode="reflect")
        distorted_image = distorted_image.reshape(image.shape)
        return distorted_image
    @staticmethod
    def apply_all_augmentations(image, et_small_alpha, et_small_sigma, et_large_alpha, et_large_sigma):
        image = Image2Augment.apply_awgn(image)
        image = Image2Augment.apply_elastic_transform(image, et_small_alpha, et_small_sigma)
        image = Image2Augment.apply_elastic_transform(image, et_large_alpha, et_large_sigma)
        return image
    @staticmethod
    def save_augment_png(dir_path, image, augment_type):
        date_time = Util.get_datetime()
        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(f"{dir_path}/{augment_type}_{date_time}.png", image)
    @staticmethod
    def resizeimg(args, img):
        h, w = img.shape
        ratio = min(args.max_width / w, args.max_height / h)
        resized_image = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        if resized_image.shape[0] > args.max_height:
            ratio = args.max_height / resized_image.shape[0]
            resized_image = cv2.resize(resized_image, (int(resized_image.shape[1] * ratio), args.max_height))
        img = resized_image
        top_pad = (args.max_height - img.shape[0]) // 2
        bottom_pad = args.max_height - img.shape[0] - top_pad
        left_pad = (args.max_width - img.shape[1]) // 2
        right_pad = args.max_width - img.shape[1] - left_pad
        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="constant", constant_values=255)
        return img

class Annotation2Xml:
    NOTE = "note"
    REST = "rest"
    BARLINE = "barline"
    DIVISION_CHORD = "+"
    DIVISION_NOTE = "|"
    DIVISION_DURATION = "_"
    DIVISION_PITCH = "-"
    NOTEHEAD_X_LIST = ("G5", "A5", "D4")
    STEM_DIRECTION_UP = "up"
    IS_NOTE = "is_note"
    NOTEHEAD = "notehead"
    PITCH = "pitch"
    DURATION = "duration"
    DURATION_TYPE_TO_LENGTH_TEMP = {
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eighth": 0.5,
        "16th": 0.25,
        "32nd": 0.125,
        "64th": 0.0625,
        "128th": 0.03125,
        "256th": 0.015625,
        "512th": 0.0078125,
        "1024th": 0.00390625,
        "breve": 8.0,
        "longa": 16.0,
        "maxima": 32.0,
    }
    DURATION_TYPE_TO_LENGTH = {}
    for duration_type, quarter_length in DURATION_TYPE_TO_LENGTH_TEMP.items():
        DURATION_TYPE_TO_LENGTH[duration_type] = quarter_length
        DURATION_TYPE_TO_LENGTH[duration_type + "."] = quarter_length + quarter_length / 2
    @staticmethod
    def fit_annotation_bar(annotation_dict):
        result_annotation = []
        total_duration_4_4 = 4.0
        for bar in annotation_dict:
            sum_duration = 0
            chord_annotation = []
            for chord_info in bar:
                first_duration = chord_info[0][Annotation2Xml.DURATION]
                if sum_duration + first_duration > total_duration_4_4:
                    break
                chord_annotation.append(chord_info)
                sum_duration += first_duration
            rest_duration = total_duration_4_4 - sum_duration
            if rest_duration > 0:
                chord_annotation.append([{Annotation2Xml.DURATION: rest_duration, Annotation2Xml.IS_NOTE: False}])
            result_annotation.append(chord_annotation)
        return result_annotation
    @staticmethod
    def m21_score_to_xml_tree(m21_score):
        musicxml_string = m21ToXml.GeneralObjectExporter(m21_score).parse()
        xml_tree = ET.ElementTree(ET.fromstring(musicxml_string))
        return xml_tree
    @staticmethod
    def split_annotation(annotation):
        annotation_dict_list = []
        bar_list = annotation.split(Annotation2Xml.BARLINE)
        for bar_info in bar_list:
            if bar_info == "":
                continue
            chord_list = bar_info.split(Annotation2Xml.DIVISION_CHORD)
            annotation_chord_list = []
            for chord_info in chord_list:
                if chord_info == "" or (chord_info[0:4] != Annotation2Xml.NOTE and chord_info[0:4] != Annotation2Xml.REST):
                    continue
                note_list = chord_info.split(Annotation2Xml.DIVISION_NOTE)
                annotation_note_list = []
                for note_info in note_list:
                    if note_info == "":
                        continue
                    note_info_dict = {}
                    pitch_info, duration = note_info.split(Annotation2Xml.DIVISION_DURATION)
                    pitch_info_list = pitch_info.split(Annotation2Xml.DIVISION_PITCH)
                    note_info_dict[Annotation2Xml.DURATION] = Annotation2Xml.DURATION_TYPE_TO_LENGTH[duration]
                    note_info_dict[Annotation2Xml.IS_NOTE] = pitch_info_list[0] == Annotation2Xml.NOTE
                    if pitch_info_list[0] == Annotation2Xml.NOTE:
                        note_info_dict[Annotation2Xml.IS_NOTE] = True
                        note_info_dict[Annotation2Xml.PITCH] = pitch_info_list[1]
                        note_info_dict[Annotation2Xml.NOTEHEAD] = None
                        if pitch_info_list[1] in Annotation2Xml.NOTEHEAD_X_LIST:
                            note_info_dict[Annotation2Xml.NOTEHEAD] = "x"
                    annotation_note_list.append(note_info_dict)
                annotation_chord_list.append(annotation_note_list)
            annotation_dict_list.append(annotation_chord_list)
        return annotation_dict_list
    @staticmethod
    def annotation_to_musicxml(annotation):
        annotation_dict = Annotation2Xml.split_annotation(annotation)
        annotation_dict = Annotation2Xml.fit_annotation_bar(annotation_dict)
        score = stream.Score()
        drum_track = stream.Part()
        drum_track.append(instrument.Percussion())
        drum_track.append(clef.PercussionClef())
        drum_track.append(meter.TimeSignature("4/4"))
        for bar in annotation_dict:
            for chord_info in bar:
                chord_notes = []
                is_note = any(item[Annotation2Xml.IS_NOTE] for item in chord_info)
                if not is_note:
                    r = note.Rest()
                    r.duration.quarterLength = chord_info[0][Annotation2Xml.DURATION]
                    drum_track.append(r)
                    continue
                for note_info in chord_info:
                    if note_info[Annotation2Xml.IS_NOTE]:
                        n = note.Unpitched(displayName=note_info[Annotation2Xml.PITCH])
                        n.duration.quarterLength = note_info[Annotation2Xml.DURATION]
                        n.stemDirection = Annotation2Xml.STEM_DIRECTION_UP
                        if note_info[Annotation2Xml.NOTEHEAD] is not None:
                            n.notehead = note_info[Annotation2Xml.NOTEHEAD]
                        chord_notes.append(n)
                chord = percussion.PercussionChord(chord_notes)
                chord.stemDirection = Annotation2Xml.STEM_DIRECTION_UP
                drum_track.append(chord)
        score.insert(0, drum_track)
        plt.clf()
        score.show()
        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        IMAGE_PATH = "../images"
        os.makedirs(IMAGE_PATH, exist_ok=True)
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{IMAGE_PATH}/output-{date_time}.png")
        xml_tree = Annotation2Xml.m21_score_to_xml_tree(score)
        return xml_tree

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # 모델이 training 시, self.add_loss()를 사용하여 loss를 계산하고 더해줌
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # test시에는 예측값만 반환
        return y_pred

class DDMOMR:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def build_model(self):
        # Inputs 정의
        input_img = layers.Input(
            shape=(self.args.max_width, self.args.max_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # 첫번째 convolution block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # 두번째 convolution block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # 앞에 2개의 convolution block에서 maxpooling(2,2)을 총 2번 사용
        # feature map의 크기는 1/4로 downsampling 됨
        # 마지막 레이어의 filter 수는 64개 다음 RNN에 넣기 전에 reshape 해줌
        new_shape = ((self.args.max_width // 4), (self.args.max_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(
            len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
        )(x)

        # 위에서 지정한 CTCLayer 클래스를 이용해서 ctc loss를 계산
        output = CTCLayer(name="ctc_loss")(labels, x)

        # 모델 정의
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()

        # 모델 컴파일
        model.compile(optimizer=opt)
        return model

class StaffToScore:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.ctc = CTCLayer()
        self.prediction_model = self.load_predict_model()
        self.model_obj = DDMOMR(args)
    def save(self, model):
        date_time = Util.get_datetime()
        os.makedirs(self.args.filepaths.model_path.base, exist_ok=True)
        model_path = f"{self.args.filepaths.model_path.model}_{date_time}.h5"
        model.save(model_path, save_format="tf")
        print("--! save model: ", model_path)
    def load_predict_model(self, model_file=None):
        model_files = glob.glob(f"{self.args.filepaths.model_path.model}_*.h5")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return
        model_files.sort(reverse=True)
        load_model_file = model_files[0]
        if model_file is not None:
            load_model_file = model_file
        print("-- ! load model: ", load_model_file)
        custom_objects = {"CTCLayer": self.ctc}
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(load_model_file, custom_objects=custom_objects)
        prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
        return prediction_model
    def load_model(self, model_file=None):
        model_files = glob.glob(f"{self.args.filepaths.model_path.model}_*.h5")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return
        model_files.sort(reverse=True)
        load_model_file = model_files[0]
        if model_file is not None:
            load_model_file = model_file
        print("-- ! load model: ", load_model_file)
        custom_objects = {"CTCLayer": self.ctc}
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(load_model_file, custom_objects=custom_objects)
        return model
    def load_x_y(self, title_path):
        title_dataset_path = [item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)]
        x_exp = re.compile(r"^[^._].*\.png$")
        y_exp = re.compile(r"^[^._].*\.txt$")
        x_raw_path_list = []
        y_raw_path_list = []
        for tdp in title_dataset_path:
            files = os.listdir(tdp)
            x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
            y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
            if len(x_raw_path) == len(y_raw_path):
                x_raw_path_list += x_raw_path
                y_raw_path_list += y_raw_path
        Util.print_step("load data")
        print(f"-- x: {len(x_raw_path_list)} | y: {len(y_raw_path_list)}")
        return x_raw_path_list, y_raw_path_list
    def load_data_path(self):
        dataset_path = f"{self.args.filepaths.feature_path.seq}/"
        title_path_list = Util.get_all_subfolders(dataset_path)
        x_raw_path_list = []
        y_raw_path_list = []
        for title_path in title_path_list:
            x_raw_path, y_raw_path = self.load_x_y(title_path)
            x_raw_path_list += x_raw_path
            y_raw_path_list += y_raw_path
        Util.print_step("load data")
        print(f"-- x: {len(x_raw_path_list)} | y: {len(y_raw_path_list)}")
        return x_raw_path_list, y_raw_path_list
    def encode_single_sample(self, img, label):
        if img.shape.ndim == 2:
            img = tf.expand_dims(img, axis=-1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label_r = char_to_num(tf.strings.split(label))
        return {"image": img, "label": label_r}
    def pitch_encode_x(self, img):
        if img.shape.ndims == 2:
            img = tf.expand_dims(img, axis=-1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}
    def load_data(self):
        x_raw_path_list, y_raw_path_list = self.load_data_path()
        x_preprocessed_list = []
        y_preprocessed_list = []
        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            y_raw_path = y_raw_path_list[idx]
            x_preprocessed = self.preprocessing(x_raw_path)
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")
            for _, img in x_preprocessed:
                x_preprocessed_list.append(img)
                y_preprocessed_list.append(y_preprocessed)
        print("전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list))
        result_note = self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)
        return x_preprocessed_list, result_note
    def load_test_data(self):
        test_path = f"{self.args.filepaths.test_path}/"
        x_raw_path_list, y_raw_path_list = self.load_x_y(test_path)
        x_preprocessed_list = []
        y_preprocessed_list = []
        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            y_raw_path = y_raw_path_list[idx]
            print(">>>>>>>>>>>>>>>>>>>>", x_raw_path)
            biImg = Image2Augment.readimg(x_raw_path)
            biImg = 255 - biImg
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")
            y_preprocessed_list.append(y_preprocessed)
        print("전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list))
        result_note = self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)
        return x_preprocessed_list, result_note
    def model_predict(self, x_preprocessed_list):
        pitch_dataset = tf.data.Dataset.from_tensor_slices(np.array(x_preprocessed_list))
        pitch_dataset = pitch_dataset.map(self.pitch_encode_x, num_parallel_calls=tf.data.AUTOTUNE).batch(self.args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        all_pred_texts = []
        all_images = []
        for batch in pitch_dataset:
            batch_images = batch["image"]
            preds = self.prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)
            all_pred_texts.extend(pred_texts)
            all_images.extend(batch_images)
        b_l = min(len(pred_texts), 10)
        _, ax = plt.subplots(b_l, 1, figsize=(24, 20))
        if b_l == 1:
            ax = [ax]
        for i in range(b_l):
            img = np.array(all_images[i])
            img = np.squeeze(img)
            img = np.transpose(img, (1, 0))
            img = img * 255
            img = img.astype(np.uint8)
            ax[i].imshow(img, cmap="gray")
            ax[i].set_title(f"Prediction: {all_pred_texts[i]}")
            ax[i].axis("off")
        os.makedirs("predict-result/", exist_ok=True)
        dt = Util.get_datetime()
        plt.savefig(f"predict-result/pred-{dt}.png")
        plt.show()
        return all_pred_texts
    def training(self):
        x, y = self.load_data()
        from sklearn.model_selection import train_test_split
        pitch_x_train, pitch_x_valid, pitch_y_train, pitch_y_valid = train_test_split(np.array(x), np.array(y), test_size=0.1, random_state=40)
        pitch_train_dataset = tf.data.Dataset.from_tensor_slices((pitch_x_train, pitch_y_train))
        pitch_train_dataset = pitch_train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(self.args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        pitch_validation_dataset = tf.data.Dataset.from_tensor_slices((pitch_x_valid, pitch_y_valid))
        pitch_validation_dataset = pitch_validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(self.args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        _, ax = plt.subplots(4, 1, figsize=(24, 20))
        for batch in pitch_train_dataset.take(1):
            images = batch["image"]
            labels = batch["label"]
            print(">>> 데이터셋 랜덤 확인")
            for i in range(4):
                img = (images[i] * 255).numpy().astype("uint8")
                label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
                ax[i].imshow(img[:, :, 0].T, cmap="gray")
                ax[i].set_title(label)
                ax[i].axis("off")
        os.makedirs(f"dataset-output/", exist_ok=True)
        dt = Util.get_datetime()
        plt.savefig(f"dataset-output/dataset-{dt}.png")
        plt.show()
        model = self.model_obj.build_model()
        model.summary()
        epochs = self.args.epoch
        early_stopping_patience = 10
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
        model.fit(pitch_train_dataset, validation_data=pitch_validation_dataset, epochs=epochs, callbacks=[early_stopping], batch_size=self.args.batch_size)
        self.save(model)
        prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
        for batch in pitch_validation_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]
            print("-- 예측 --")
            preds = prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)
            orig_texts = []
            y_true = []
            print("-- 실제 정답 --")
            for label in batch_labels:
                y_true.append(label.numpy().tolist())
                label = tf.strings.join(num_to_char(label), separator=" ").numpy().decode("utf-8").replace("[UNK]", "")
                orig_texts.append(label)
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            ser = self.symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            print("Symbol Error Rate:", ser.numpy())
            b_l = len(pred_texts)
            _, ax = plt.subplots(b_l, 1)
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                title_true = f"Ground Truth: {orig_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_true}\n{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"test-result/", exist_ok=True)
        plt.savefig(f"test-result/pred.png")
        plt.show()
    def load_prediction_model(self, checkpoint_path):
        model = self.model_obj.build_model()
        check = tf.train.latest_checkpoint(checkpoint_path)
        print("-- Loading weights from:", check)
        prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
        prediction_model.load_weights(check).expect_partial()
        prediction_model.summary()
        return prediction_model
    def pitch_decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.args.max_seq_len]
        output_text = []
        y_pred = []
        for res in results:
            y_pred.append(res.numpy().tolist())
            res = tf.strings.join(num_to_char(res), separator=" ").numpy().decode("utf-8").replace("[UNK]", "").rstrip("+")
            output_text.append(res)
        return output_text, y_pred
    def test(self):
        x, y = self.load_test_data()
        pitch_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        pitch_dataset = pitch_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(self.args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        for batch in pitch_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]
            print("-- 예측 --")
            preds = self.prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)
            orig_texts = []
            y_true = []
            print("-- 실제 정답 --")
            for label in batch_labels:
                y_true.append(label.numpy().tolist())
                label = tf.strings.join(num_to_char(label), separator=" ").numpy().decode("utf-8").replace("[UNK]", "")
                orig_texts.append(label)
            b_l = min(len(pred_texts), 10)
            _, ax = plt.subplots(b_l, 1, figsize=(24, 20))
            for i in range(b_l):
                img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img = img.T
                title_pred = f"Prediction: {pred_texts[i]}"
                title_true = f"Ground Truth: {orig_texts[i]}"
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"{title_true}\n{title_pred}")
                ax[i].axis("off")
        os.makedirs(f"predict-result/", exist_ok=True)
        dt = Util.get_datetime()
        plt.savefig(f"predict-result/pred-{dt}.png")
        plt.show()
    def preprocessing(self, rgb):
        augment_result = Image2Augment.process_image2augment(self.args, rgb)
        resize_result = []
        for typename, img in augment_result:
            resizeimg = Image2Augment.resizeimg(self.args, img)
            resize_result.append((typename, resizeimg))
        return resize_result
    def map_notes2pitch_rhythm_lift_note(self, note_list):
        result_note = []
        for notes in note_list:
            group_notes_token_len = 0
            group_note = []
            note_split = notes.split("+")
            for note_s in note_split:
                if "|" in note_s:
                    note_split_chord = note_s.split("|")
                    group_note.append(" | ".join(note_split_chord))
                    group_notes_token_len += (len(note_split_chord) + len(note_split_chord) - 1)
                else:
                    group_note.append(note_s)
                    group_notes_token_len += 1
            emb_note = " ".join(group_note)
            toks_len = group_notes_token_len
            if toks_len < self.args.max_seq_len:
                for _ in range(self.args.max_seq_len - toks_len):
                    emb_note += " [PAD]"
            result_note.append(emb_note)
        return result_note
    def symbol_error_rate(self, y_true, y_pred):
        padding_indices = tf.where(tf.equal(y_true, -1))
        padding_values = tf.zeros_like(padding_indices[:, 0], dtype=tf.int32)
        y_true = tf.tensor_scatter_nd_update(y_true, padding_indices, padding_values)
        y_pred = tf.tensor_scatter_nd_update(y_pred, padding_indices, padding_values)
        errors = tf.not_equal(y_true, y_pred)
        ser = tf.reduce_mean(tf.cast(errors, tf.float32))
        return ser

class SheetToScore:
    def __init__(self, args):
        self.args = args
        self.staff2score = StaffToScore(args)
    def extract_segment_from_score(self, biImg):
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(biImg)
        return cnt, labels, stats, centroids
    def extract_stave_from_score(self, biImg, cnt, stats):
        score_h, score_w = biImg.shape
        PAD = 10
        stave_list = []
        for i in range(1, cnt):
            x_s, y_s, w_s, h_s, _ = stats[i]
            x = round(x_s - PAD)
            t_y = round((y_s - PAD))
            y = max(t_y, 0)
            h = h_s + 2 * PAD
            w = round(w_s + 2 * PAD)
            if w >= score_w * 0.3:
                stave = biImg[y : y + h, x : x + w]
                stave_list.append(stave)
        result_stave_list = []
        for stave in stave_list:
            h, _ = stave.shape
            if h >= 10:
                result_stave_list.append(stave)
        return result_stave_list
    def save_stave(self, title, stave_list):
        os.makedirs(f"{self.args.filepaths.feature_path.base}/stave/{title}", exist_ok=True)
        for idx, stave in enumerate(stave_list):
            date_time = Util.get_datetime()
            cv2.imwrite(f"{self.args.filepaths.feature_path.base}/stave/{title}/{title}-stave_{idx+1}_{date_time}.png", stave)
            print(idx + 1, "--shape: ", stave.shape)
    def transform_score2stave(self, score_path):
        biImg = Image2Augment.readimg(score_path)
        (h, w) = biImg.shape[:2]
        target_width = self.args.score_width
        ratio = target_width / float(w)
        target_height = int(h * ratio)
        biImg = cv2.resize(biImg, (target_width, target_height), interpolation=cv2.INTER_AREA)
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list
    def transform_scoreImg2stave(self, score):
        biImg = Image2Augment.img2binary(score)
        (h, w) = biImg.shape[:2]
        target_width = self.args.score_width
        ratio = target_width / float(w)
        target_height = int(h * ratio)
        biImg = cv2.resize(biImg, (target_width, target_height), interpolation=cv2.INTER_AREA)
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list
    def stave2measure(self, stave):
        h, w = stave.shape
        max_h = self.args.max_height
        max_w = self.args.max_width
        new_width = int((max_h / h) * w)
        resized_stave = cv2.resize(stave, (new_width, max_h))
        result = []
        start_x = 0
        _, r_w = resized_stave.shape
        while start_x < r_w:
            end_x = min(start_x + max_w, r_w)
            cropped_image = resized_stave[:, start_x:end_x]
            if end_x - start_x < max_w:
                padding_needed = max_w - (end_x - start_x)
                cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, 0, padding_needed, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            result.append(255 - cropped_image)
            start_x += max_w
        return result
    def preprocessing(self, score_path):
        stave_list = self.transform_score2stave(score_path)
        measure_list = []
        for idx, stave in enumerate(stave_list):
            measures = self.stave2measure(stave)
            measure_list += measures
        x_preprocessed_list = []
        print("measure_list>>>>>>>>>", measure_list)
        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))
        return x_preprocessed_list
    def imagePreprocessing(self, score):
        stave_list = self.transform_scoreImg2stave(score)
        measure_list = []
        for idx, stave in enumerate(stave_list):
            measures = self.stave2measure(stave)
            measure_list += measures
        x_preprocessed_list = []
        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))
        return x_preprocessed_list
    def postprocessing(self, predict_result):
        def process_string(s):
            s = re.sub(r"\s*\|\s*", "|", s)
            s = re.sub(r"\s+", "+", s)
            if s[-1] == "+":
                s = s[:-1]
            return s
        result_list = []
        for res in predict_result:
            result_list.append(process_string(res))
        result = "+".join(result_list)
        print(">>>>", result)
        return result
    def predict(self, score_path):
        x_preprocessed_list = self.preprocessing(score_path)
        result = self.staff2score.model_predict(x_preprocessed_list)
        postresult = self.postprocessing(result)
        return postresult
    def inferSheetToXml(self, score):
        x_preprocessed_list = self.imagePreprocessing(score)
        result = self.staff2score.model_predict(x_preprocessed_list)
        postresult = self.postprocessing(result)
        xml_tree = Annotation2Xml.annotation_to_musicxml(postresult)
        return xml_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference single staff image")
    parser.add_argument("filepath", type=str, help="path to staff image")
    # Provide a default filepath when running in a Jupyter Notebook
    # Replace 'your_file_path.png' with the actual path to your image.
    parsed_args = parser.parse_args(['drum_sheet_music.png'])
    cofigpath = f"letmedrum\config.yaml"
    # ddm-omr/workspace/config.yaml
    args = getconfig(cofigpath)
    score_path = parsed_args.filepath
    handler = SheetToScore(args)
    predict_result = handler.predict(score_path)
    print(predict_result)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Inference single staff image")
#     parser.add_argument("filepath", type=str, help="path to staff image")
#     parsed_args = parser.parse_args()
#     cofigpath = f"ddm-omr/workspace/config.yaml"
#     args = getconfig(cofigpath)
#     score_path = parsed_args.filepath
#     handler = SheetToScore(args)
#     predict_result = handler.predict(score_path)
#     print(predict_result)