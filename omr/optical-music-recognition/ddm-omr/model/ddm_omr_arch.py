from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

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
