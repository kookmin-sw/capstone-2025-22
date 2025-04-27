import os
import cv2
import numpy as np
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

from constant.common import (
    ALL_AUGMENT,
    AUGMENT,
    AWGN,
    ET_LARGE,
    ET_SMALL,
    EXP,
    MULTI_LABEL,
    ORIGIN,
    PNG,
)
from constant.path import DATA_FEATURE_PATH, DATA_RAW_PATH, OSMD
from score2stave import Score2Stave
from util import Util

# Set parameters
et_small_alpha_training = 8
et_small_sigma_evaluation = 4

et_large_alpha_training = 1000
et_large_sigma_evaluation = 80


class Image2Augment:
    def process_all_image2augment():
        """
        raw/osmd-v1 에 있는 score를 augment해서 processed-feature/augment에 저장
        """
        # osmd title paths 가져오기
        title_path_ = f"{DATA_RAW_PATH}/{OSMD}"
        title_path_list = Util.get_all_subfolders(title_path_)
        # title 마다 score2stave -- score가 한 장이라는 전제
        for title_path in title_path_list:
            # 모든 score 불러와서 score2stave 후, padding 준 거 저장
            score_path_list = Util.get_all_files(f"{title_path}", EXP[PNG])
            for score_path in score_path_list:
                Image2Augment.process_image2augment(score_path)

    def process_image2augment(score_path):
        # -- score -> stave
        title = Util.get_title(score_path)
        input_image = Score2Stave.transform_img2binaryImg(score_path)
        input_image = 255 - input_image

        # Apply augmentations
        awgn_image = Image2Augment.apply_awgn(input_image)

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
            (ORIGIN, input_image),
            (AWGN, awgn_image),
            (ET_SMALL, et_small_image),
            (ET_LARGE, et_large_image),
            (ALL_AUGMENT, all_augmentations_image),
        ]

        for name, img in result:
            Image2Augment.save_augment_png(title, img, MULTI_LABEL, name)

    @staticmethod
    def apply_awgn(image, sigma=0.1):
        noisy_image = random_noise(image, mode="gaussian", var=sigma**2)
        return (255 * noisy_image).astype(np.uint8)

    @staticmethod
    def apply_elastic_transform(image, alpha, sigma):
        # Elastic Transformations
        random_state = np.random.RandomState(None)
        height, width = image.shape[:2]
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # 좌표 배열을 생성하여 좌표에 변환을 적용
        distorted_x = np.clip(x + dx, 0, width - 1)
        distorted_y = np.clip(y + dy, 0, height - 1)

        print("--", distorted_x.shape)
        print("--", distorted_y.shape)

        distorted_image = map_coordinates(
            image,
            [distorted_y, distorted_x],
            order=1,
            mode="reflect",
        )
        distorted_image = distorted_image.reshape(image.shape)
        return distorted_image

    @staticmethod
    def apply_all_augmentations(
        image,
        et_small_alpha,
        et_small_sigma,
        et_large_alpha,
        et_large_sigma,
    ):
        image = Image2Augment.apply_awgn(image)
        image = Image2Augment.apply_elastic_transform(
            image, et_small_alpha, et_small_sigma
        )
        image = Image2Augment.apply_elastic_transform(
            image, et_large_alpha, et_large_sigma
        )
        return image

    @staticmethod
    def save_augment_png(title, image, label_type, state):
        """
        save AUGMENT png
        """
        folder_path = f"{DATA_FEATURE_PATH}/{label_type}/{title}/{AUGMENT}"
        os.makedirs(folder_path, exist_ok=True)
        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{folder_path}/{title}_{state}_{date_time}.{EXP[PNG]}",
            image,
        )
        print(state, "--AUGMENT shape: ", image.shape)
