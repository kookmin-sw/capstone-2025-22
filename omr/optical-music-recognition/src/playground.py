import cv2
from matplotlib import pyplot as plt
from constant.common import AUGMENT, EXP, MULTI_LABEL, NOTE, PNG, XML
from constant.path import DATA_TEST_PATH, IMAGE_PATH, RESULT_XML_PATH
from model.multilabel_note_model import MultiLabelNoteModel
from feature_labeling import FeatureLabeling
from data_processing import DataProcessing
from Image2augment import Image2Augment
from model.tromr_arch import TrOMR
from score2stave import Score2Stave
from note2xml import Note2XML
from util import Util

import albumentations as alb
from albumentations.pytorch import ToTensorV2


# ======================== note2xml ===========================
def create_musicxml_dummy():
    note_data = {
        "attributes": {"divisions": 32, "beats": 4, "beat-type": 4},
        "notes": [
            {"step": "F", "octave": 4, "duration": 32, "type": "quarter"},
            {
                "step": "G",
                "octave": 5,
                "duration": 32,
                "type": "quarter",
                "chord": True,
            },
            {"duration": 32, "type": "quarter"},
            {"step": "A", "octave": 4, "duration": 32, "type": "quarter"},
        ],
    }

    # Create ElementTree and write to file
    datetime = Util.get_datetime()
    Note2XML.create_musicxml(
        note_data, f"{RESULT_XML_PATH}/musicxml-{datetime}.{EXP[XML]}"
    )


# ======================== image augmentation ===========================
def process_image2augment():
    title = "Rock-ver"
    # Example usage
    input_image = Score2Stave.transform_img2binaryImg(f"{title}.png")
    input_image = 255 - input_image

    # Set parameters
    et_small_alpha_training = 8
    et_small_sigma_evaluation = 4

    et_large_alpha_training = 1000
    et_large_sigma_evaluation = 80

    # Apply augmentations
    awgn_image = Image2Augment.apply_awgn(input_image)

    # apn_image = Image2Augment.apply_apn(input_image)

    et_small_image = Image2Augment.apply_elastic_transform(
        input_image, et_small_alpha_training, et_small_sigma_evaluation
    )
    et_large_image = Image2Augment.apply_elastic_transform(
        input_image, et_large_alpha_training, et_large_sigma_evaluation
    )
    all_augmentations_image = Image2Augment.apply_all_augmentations(
        input_image,
        et_small_alpha_training,
        et_small_sigma_evaluation,
        et_large_alpha_training,
        et_large_sigma_evaluation,
    )

    result = [
        ("awgn_image", awgn_image),
        # ("apn_image", apn_image),
        ("et_small_image", et_small_image),
        ("et_large_image", et_large_image),
        ("all_augmentations_image", all_augmentations_image),
    ]

    for name, img in result:
        date_time = Util.get_datetime()
        output_path = f"{IMAGE_PATH}/{AUGMENT}/{title}-{name}-{date_time}.png"
        cv2.imwrite(output_path, img)


# process_image2augment()

channels = 1
patch_size = 16
max_height = 128
max_width = 1280
max_seq_len = 256
transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # ToTensorV2(),
    ]
)


def preprocessing(rgb):
    size_h = max_height

    h, w, c = rgb.shape
    new_h = size_h
    new_w = int(size_h / h * w)
    new_w = new_w // patch_size * patch_size
    img = cv2.resize(rgb, (new_w, new_h))
    img = transform(image=img)["image"][:1]
    date_time = Util.get_datetime()
    cv2.imwrite(f"{DATA_TEST_PATH}/test-{date_time}.png", img)

    return img


def predict_token(imgpath):
    imgs = readimg(imgpath)
    preprocessing(imgs)
    # imgs = torch.cat(imgs).float().unsqueeze(1)
    # output = model.generate(imgs.to(device), temperature=args.get("temperature", 0.2))
    # rhythm, pitch, lift = output
    # return rhythm, pitch, lift


def predict_img2token(rgbimgs):
    if not isinstance(rgbimgs, list):
        rgbimgs = [rgbimgs]
    imgs = [preprocessing(item) for item in rgbimgs]
    # imgs = torch.cat(imgs).float().unsqueeze(1)
    # output = model.generate(imgs.to(device), temperature=args.get("temperature", 0.2))
    # rhythm, pitch, lift = output
    # return rhythm, pitch, lift


def readimg(path):

    size_h = max_height

    img = cv2.imread(path)
    print(1)
    print(img.shape)

    # if img.shape[-1] == 4:
    #     img = 255 - img[:, :, 3]
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # elif img.shape[-1] == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # else:
    #     raise RuntimeError("Unsupport image type!")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    new_h = size_h
    new_w = int(size_h / h * w)
    new_w = new_w // patch_size * patch_size
    img = cv2.resize(img, (new_w, new_h))
    # img = transform(image=img)["image"]
    print(2)
    print(img.shape)
    print(img.dtype)
    cv2.imwrite(f"{DATA_TEST_PATH}/test-02.png", img)
    return img


# readimg(f"{DATA_TEST_PATH}/test-01.png")
# predict_token(f"{DATA_TEST_PATH}/test-01.png")
