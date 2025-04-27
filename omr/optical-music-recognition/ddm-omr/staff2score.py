import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
from sklearn.model_selection import train_test_split

from process_data.image2augment import Image2Augment
from util import Util
from model.ddm_omr_arch import DDMOMR, CTCLayer


# # GPU:1만 사용하도록 환경 변수 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # 선택된 GPU 확인 및 메모리 증가 설정
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     try:
#         # GPU 1의 메모리 증가 설정
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         logical_gpus = tf.config.experimental.list_logical_devices("GPU")
#         print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
#     except RuntimeError as e:
#         print(e)


char_to_int_mapping = [
    "|",  # 0
    "barline",  # 1
    "clef-percussion",  # 2
    "timeSignature-4/4",  # 3
    # 1
    # "note-D4_eighth",
    # "note-D4_eighth.",
    # "note-D4_half",
    # "note-D4_half.",
    # "note-D4_quarter",
    # "note-D4_quarter.",
    # "note-D4_16th",
    # "note-D4_16th.",
    # "note-D4_whole",
    # "note-D4_whole.",
    # "note-D4_32nd",
    # # 2
    # "note-E4_eighth",
    # "note-E4_eighth.",
    # "note-E4_half",
    # "note-E4_half.",
    # "note-E4_quarter",
    # "note-E4_quarter.",
    # "note-E4_16th",
    # "note-E4_16th.",
    # "note-E4_whole",
    # "note-E4_whole.",
    # "note-E4_32nd",
    # 3 ------------------ kick
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
    # # 4
    # "note-G4_eighth",
    # "note-G4_eighth.",
    # "note-G4_half",
    # "note-G4_half.",
    # "note-G4_quarter",
    # "note-G4_quarter.",
    # "note-G4_16th",
    # "note-G4_16th.",
    # "note-G4_whole",
    # "note-G4_whole.",
    # "note-G4_32nd",
    # # 5
    # "note-A4_eighth",
    # "note-A4_eighth.",
    # "note-A4_half",
    # "note-A4_half.",
    # "note-A4_quarter",
    # "note-A4_quarter.",
    # "note-A4_16th",
    # "note-A4_16th.",
    # "note-A4_whole",
    # "note-A4_whole.",
    # "note-A4_32nd",
    # # 6
    # "note-B4_eighth",
    # "note-B4_eighth.",
    # "note-B4_half",
    # "note-B4_half.",
    # "note-B4_quarter",
    # "note-B4_quarter.",
    # "note-B4_16th",
    # "note-B4_16th.",
    # "note-B4_whole",
    # "note-B4_whole.",
    # "note-B4_32nd",
    # 7 ------------------ snare
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
    # # 8
    # "note-D5_eighth",
    # "note-D5_eighth.",
    # "note-D5_half",
    # "note-D5_half.",
    # "note-D5_quarter",
    # "note-D5_quarter.",
    # "note-D5_16th",
    # "note-D5_16th.",
    # "note-D5_whole",
    # "note-D5_whole.",
    # "note-D5_32nd",
    # # 9
    # "note-E5_eighth",
    # "note-E5_eighth.",
    # "note-E5_half",
    # "note-E5_half.",
    # "note-E5_quarter",
    # "note-E5_quarter.",
    # "note-E5_16th",
    # "note-E5_16th.",
    # "note-E5_whole",
    # "note-E5_whole.",
    # "note-E5_32nd",
    # # 10
    # "note-F5_eighth",
    # "note-F5_eighth.",
    # "note-F5_half",
    # "note-F5_half.",
    # "note-F5_quarter",
    # "note-F5_quarter.",
    # "note-F5_16th",
    # "note-F5_16th.",
    # "note-F5_whole",
    # "note-F5_whole.",
    # "note-F5_32nd",
    # 11 ------------------ hihat
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
    # # 12
    # "note-A5_eighth",
    # "note-A5_eighth.",
    # "note-A5_half",
    # "note-A5_half.",
    # "note-A5_quarter",
    # "note-A5_quarter.",
    # "note-A5_16th",
    # "note-A5_16th.",
    # "note-A5_whole",
    # "note-A5_whole.",
    # "note-A5_32nd",
    # # 13
    # "note-B5_eighth",
    # "note-B5_eighth.",
    # "note-B5_half",
    # "note-B5_half.",
    # "note-B5_quarter",
    # "note-B5_quarter.",
    # "note-B5_16th",
    # "note-B5_16th.",
    # "note-B5_whole",
    # "note-B5_whole.",
    # "note-B5_32nd",
    # #
    # "rest_eighth",  # 13
    # "rest_eighth.",  # 14
    # "rest_half",  # 15
    # "rest_half.",  # 16
    # "rest_quarter",  # 17
    # "rest_quarter.",  # 18
    # "rest_16th",  # 19
    # "rest_16th.",  # 20
    # "rest_whole",  # 21
    # "rest_whole.",  # 22
    # "rest_32nd",  # 23
]

# 문자를 숫자로 변환
char_to_num = layers.StringLookup(vocabulary=list(char_to_int_mapping), mask_token=None)

# 숫자를 문자로 변환
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


class StaffToScore(object):
    def __init__(self, args):
        self.args = args
        self.model = DDMOMR(args)
        self.ctc = CTCLayer()
        self.prediction_model = self.load_predict_model()

    def save(self, model):
        """
        학습한 모델 저장
        """
        date_time = Util.get_datetime()
        os.makedirs(self.args.filepaths.model_path.base, exist_ok=True)
        model_path = f"{self.args.filepaths.model_path.model}_{date_time}.h5"
        model.save(model_path, save_format="tf")
        print("--! save model: ", model_path)

    def load_predict_model(self, model_file=None):
        """
        -- method_type과 feature type에 맞는 가장 최근 모델 불러오기
        """
        model_files = glob.glob(f"{self.args.filepaths.model_path.model}_*.h5")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        load_model_file = model_files[0]  # 가장 최근 모델

        if model_file is not None:
            load_model_file = model_file
        print("-- ! load model: ", load_model_file)

        # 사용자 정의 레이어를 포함하는 딕셔너리를 만듭니다
        # custom_object_scope를 사용하여 모델을 로드합니다
        custom_objects = {"CTCLayer": self.ctc}
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                load_model_file, custom_objects=custom_objects
            )
        # 예측 모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output,
        )

        return prediction_model
    
    def load_model(self, model_file=None):
        """
        -- method_type과 feature type에 맞는 가장 최근 모델 불러오기
        """
        model_files = glob.glob(f"{self.args.filepaths.model_path.model}_*.h5")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        load_model_file = model_files[0]  # 가장 최근 모델

        if model_file is not None:
            load_model_file = model_file
        print("-- ! load model: ", load_model_file)

        # 사용자 정의 레이어를 포함하는 딕셔너리를 만듭니다
        # custom_object_scope를 사용하여 모델을 로드합니다
        custom_objects = {"CTCLayer": self.ctc}
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                load_model_file, custom_objects=custom_objects
            )


        return model


    def load_x_y(self, title_path):
        """ """
        # only measure folder
        title_dataset_path = [
            item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)
        ]

        x_exp = re.compile(r"^[^._].*\.png$")  # png exp
        y_exp = re.compile(r"^[^._].*\.txt$")  # txt exp

        x_raw_path_list = []  # image
        y_raw_path_list = []  # label

        for tdp in title_dataset_path:
            files = os.listdir(tdp)
            x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
            y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
            if len(x_raw_path) == len(y_raw_path):  # 개수 맞는 지 확인하고 추가
                x_raw_path_list += x_raw_path
                y_raw_path_list += y_raw_path

        return x_raw_path_list, y_raw_path_list

    def load_data_path(self):
        """
        각 measure에 대한 (png, txt) path 가져오기
        """
        dataset_path = f"{self.args.filepaths.feature_path.seq}/"
        title_path_list = Util.get_all_subfolders(dataset_path)

        x_raw_path_list = []  # image
        y_raw_path_list = []  # label

        for title_path in title_path_list:
            x_raw_path, y_raw_path = self.load_x_y(title_path)
            x_raw_path_list += x_raw_path
            y_raw_path_list += y_raw_path

            # title_dataset_path = [  # only measure folder
            #     item for item in glob.glob(f"{title_path}/*") if os.path.isdir(item)
            # ]

            # x_exp = re.compile(r"^[^._].*\.png$")  # png exp
            # y_exp = re.compile(r"^[^._].*\.txt$")  # txt exp

            # for tdp in title_dataset_path:
            #     files = os.listdir(tdp)
            #     x_raw_path = [f"{tdp}/{file}" for file in files if x_exp.match(file)]
            #     y_raw_path = [f"{tdp}/{file}" for file in files if y_exp.match(file)]
            #     if len(x_raw_path) == len(y_raw_path):  # 개수 맞는 지 확인하고 추가

        Util.print_step("load data")
        print(f"-- x: {len(x_raw_path_list)} | y: {len(y_raw_path_list)}")

        return x_raw_path_list, y_raw_path_list

    def encode_single_sample(self, img, label):
        # 1. 이미지로 변환하고 grayscale로 변환
        if img.shape.ndims == 2:
            img = tf.expand_dims(img, axis=-1)  # 채널 추가
        # 2. [0,255]의 정수 범위를 [0,1]의 실수 범위로 변환 및 resize
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        # 3. 이미지의 가로 세로 변환
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. 라벨 값의 문자를 숫자로 변환
        label_r = char_to_num(tf.strings.split(label))

        # 7. 딕셔너리 형태로 return
        return {"image": img, "label": label_r}

    def pitch_encode_x(self, img):
        # 1. 이미지로 변환하고 grayscale로 변환
        if img.shape.ndims == 2:
            img = tf.expand_dims(img, axis=-1)  # 채널 추가
        # 2. [0,255]의 정수 범위를 [0,1]의 실수 범위로 변환 및 resize
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.args.max_height, self.args.max_width])
        # 3. 이미지의 가로 세로 변환
        img = tf.transpose(img, perm=[1, 0, 2])

        return {"image": img}

    def load_data(self):
        x_raw_path_list, y_raw_path_list = self.load_data_path()

        x_preprocessed_list = []
        y_preprocessed_list = []

        for idx in range(len(x_raw_path_list)):
            x_raw_path = x_raw_path_list[idx]
            # print(x_raw_path)
            y_raw_path = y_raw_path_list[idx]

            # augment 5개 데이터 생성됐다면 -> y도 그만큼 복제해주기
            x_preprocessed = self.preprocessing(x_raw_path)

            # annotation에서 띄어쓰기 있는 것들 사이사이는 + 로 연결해주기
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")

            for _, img in x_preprocessed:
                # print("-- resize 후 : ", img.shape)
                x_preprocessed_list.append(img)
                y_preprocessed_list.append(y_preprocessed)

        print(
            "전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list)
        )
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

            # annotation에서 띄어쓰기 있는 것들 사이사이는 + 로 연결해주기
            y_preprocessed = Util.read_txt_file(y_raw_path)
            y_preprocessed = y_preprocessed.replace(" ", "+").replace("\t", "+")
            y_preprocessed_list.append(y_preprocessed)

        print(
            "전처리 후 x, y 개수: ", len(x_preprocessed_list), len(y_preprocessed_list)
        )
        result_note = self.map_notes2pitch_rhythm_lift_note(y_preprocessed_list)
        return x_preprocessed_list, result_note

    def model_predict(self, x_preprocessed_list):
        # print(">>>>>>>>>>>", len(x_preprocessed_list))
        # 리스트를 tf.Tensor로 변환
        pitch_dataset = tf.data.Dataset.from_tensor_slices(
            np.array(x_preprocessed_list)
        )

        pitch_dataset = (
            pitch_dataset.map(self.pitch_encode_x, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        all_pred_texts = []
        all_images = []

        # 모든 배치에 대해 예측 수행
        for batch in pitch_dataset:
            batch_images = batch["image"]
            preds = self.prediction_model(batch_images)
            pred_texts, y_pred = self.pitch_decode_batch_predictions(preds)

            all_pred_texts.extend(pred_texts)
            all_images.extend(batch_images)

        # 예측 결과 시각화 및 저장
        b_l = min(len(pred_texts), 10)
        _, ax = plt.subplots(b_l, 1, figsize=(24, 20))
        # Ensure ax is always iterable
        if b_l == 1:
            ax = [ax]
        for i in range(b_l):
            img = np.array(all_images[i])  # 이미지 데이터 가져오기
            img = np.squeeze(img)  # 3차원 배열에서 불필요한 차원 제거
            img = np.transpose(img, (1, 0))  # 이미지 데이터를 전치하여 가로 세로를 바꿈
            img = img * 255  # 0-1 범위를 0-255로 변환
            img = img.astype(np.uint8)  # 이미지 데이터를 uint8 형식으로 변환

            ax[i].imshow(img, cmap="gray")
            ax[i].set_title(f"Prediction: {all_pred_texts[i]}")
            ax[i].axis("off")

        os.makedirs("predict-result/", exist_ok=True)
        datetime = Util.get_datetime()
        plt.savefig(f"predict-result/pred-{datetime}.png")
        plt.show()

        return all_pred_texts  # 예측 결과 반환

    def training(self):
        x, y = self.load_data()
        

        pitch_x_train, pitch_x_valid, pitch_y_train, pitch_y_valid = train_test_split(
            np.array(x), np.array(y), test_size=0.1, random_state=40
        )

        pitch_train_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_train, pitch_y_train)
        )
        pitch_train_dataset = (
            pitch_train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        pitch_validation_dataset = tf.data.Dataset.from_tensor_slices(
            (pitch_x_valid, pitch_y_valid)
        )
        pitch_validation_dataset = (
            pitch_validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        _, ax = plt.subplots(4, 1, figsize=(24, 20))
        for batch in pitch_train_dataset.take(1):
            images = batch["image"]
            labels = batch["label"]

            print(">>> 데이터셋 랜덤 확인")

            for i in range(4):
                img = (images[i] * 255).numpy().astype("uint8")
                label = (
                    tf.strings.reduce_join(num_to_char(labels[i]))
                    .numpy()
                    .decode("utf-8")
                )
                # label = labels[i]
                # print(labels[i])
                ax[i].imshow(img[:, :, 0].T, cmap="gray")
                ax[i].set_title(label)
                ax[i].axis("off")
        os.makedirs(f"dataset-output/", exist_ok=True)
        datetime=Util.get_datetime()
        plt.savefig(f"dataset-output/dataset-{datetime}.png")
        plt.show()

        # Get the model
        # ----------------- 이어서 학습시키고 싶으면-------------
        # model = self.load_model()
        # ----------------- 새롭게 학습시키고 싶으면-------------
        model = self.model.build_model()

        model.summary()

        epochs = self.args.epoch
        early_stopping_patience = 10
        # early stopping 지정
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )

        # Train the model with checkpointing
        model.fit(
            pitch_train_dataset,
            validation_data=pitch_validation_dataset,
            epochs=epochs,
            callbacks=[early_stopping],
            batch_size=self.args.batch_size,
        )

        # save
        self.save(model)

        # 예측 모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output,
        )

        #  validation dataset에서 하나의 배치를 시각화
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
                # print(label)
                y_true.append(label.numpy().tolist())
                label = (
                    tf.strings.join(num_to_char(label), separator=" ")
                    .numpy()
                    .decode("utf-8")
                    .replace("[UNK]", "")
                )
                orig_texts.append(label)

            # SER 계산
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            ser = self.symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            print("Symbol Error Rate:", ser.numpy())

            # _, ax = plt.subplots(4, 4, figsize=(15, 5))
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
        model = self.model.build_model()
        check = tf.train.latest_checkpoint(checkpoint_path)
        print("-- Loading weights from:", check)

        # 예측 모델 만들기
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output,
        )
        prediction_model.load_weights(check).expect_partial()
        prediction_model.summary()
        return prediction_model

    def pitch_decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[
            0
        ][0][:, : self.args.max_seq_len]
        output_text = []
        y_pred = []
        for res in results:
            # print(res)
            y_pred.append(res.numpy().tolist())
            res = (
                tf.strings.join(num_to_char(res), separator=" ")
                .numpy()
                .decode("utf-8")
                .replace("[UNK]", "")
                .rstrip("+")
            )
            output_text.append(res)
        return output_text, y_pred

    def test(self):
        x, y = self.load_test_data()

        pitch_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        pitch_dataset = (
            pitch_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.args.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        #  validation dataset에서 하나의 배치를 시각화
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
                # print(label)
                y_true.append(label.numpy().tolist())
                label = (
                    tf.strings.join(num_to_char(label), separator=" ")
                    .numpy()
                    .decode("utf-8")
                    .replace("[UNK]", "")
                )
                orig_texts.append(label)

            # SER 계산
            print(tf.constant(y_true))
            print(tf.constant(y_pred))
            # ser = symbol_error_rate(tf.constant(y_true), tf.constant(y_pred))
            # print("Symbol Error Rate:", ser.numpy())

            # _, ax = plt.subplots(4, 4, figsize=(15, 5))
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
        datetime = Util.get_datetime()
        plt.savefig(f"predict-result/pred-{datetime}.png")
        plt.show()

    def preprocessing(self, rgb):
        """
        하나의 image에 대한 전처리
        - augmentation 적용 -> binaryimage(origin), awgn,
        - model의 입력 크기에 맞게 resize & pad
        """
        augment_result = Image2Augment.process_image2augment(self.args, rgb)
        resize_result = []
        for typename, img in augment_result:
            # print("-- resize 전 : ", img.shape)
            resizeimg = Image2Augment.resizeimg(self.args, img)
            resize_result.append((typename, resizeimg))

        # # (확인용) 전처리 적용된 거 저장
        # for typename, img in augment_result:
        #     Image2Augment.save_augment_png("augment-output", img, typename)

        # # (확인용) 전처리 적용된 거 저장
        # for typename, img in resize_result:
        #     Image2Augment.save_augment_png("zeropadding-output", img, typename)

        return resize_result

    def map_notes2pitch_rhythm_lift_note(self, note_list):
        result_note = []

        for notes in note_list:
            group_notes_token_len = 0
            group_note = []

            # 우선 +로 나누고, 안에 | 있는 지 확인해서 먼저 붙이기
            # note-G#3_eighth + note-G3_eighth + note-G#3_eighth|note-G#3_eighth + rest-quarter
            note_split = notes.split("+")
            for note_s in note_split:
                if "|" in note_s:
                    mapped_note_chord = []

                    # note-G#3_eighth|note-G#3_eighth
                    # -> (note-G#3_eighth) (note-G#3_eighth)
                    note_split_chord = note_s.split("|")
                    mapped_note_chord = note_split_chord
                    group_note.append(" | ".join(mapped_note_chord))

                    # '|' 도 token이기 때문에 추가된 token 개수 더하기
                    # 동시에 친 걸 하나의 string으로 해버리는 거니까 주의하기
                    group_notes_token_len += (
                        len(note_split_chord) + len(note_split_chord) - 1
                    )
                else:
                    group_note.append(note_s)
                    group_notes_token_len += 1

            emb_note = " ".join(group_note)
            toks_len = group_notes_token_len
            # 뒤에 남은 건 패딩
            if toks_len < self.args.max_seq_len:
                for _ in range(self.args.max_seq_len - toks_len):
                    emb_note += " [PAD]"
            result_note.append(emb_note)
        return result_note

    def symbol_error_rate(self, y_true, y_pred):
        # Find indices of padding (-1)
        padding_indices = tf.where(tf.equal(y_true, -1))
        
        # Mask padding indices
        padding_values = tf.zeros_like(padding_indices[:, 0], dtype=tf.int32)
        y_true = tf.tensor_scatter_nd_update(y_true, padding_indices, padding_values)
        y_pred = tf.tensor_scatter_nd_update(y_pred, padding_indices, padding_values)
        
        # Compute symbol error rate
        errors = tf.not_equal(y_true, y_pred)
        ser = tf.reduce_mean(tf.cast(errors, tf.float32))
        
        return ser
