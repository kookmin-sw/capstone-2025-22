import os
import cv2
import numpy as np
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from util import Util


class Image2Augment:
    # def process_all_image2augment():
    #     # osmd title paths 가져오기
    #     title_path_ = f"{DATA_RAW_PATH}/{OSMD}"
    #     title_path_list = Util.get_all_subfolders(title_path_)
    #     # title 마다 score2stave -- score가 한 장이라는 전제
    #     for title_path in title_path_list:
    #         # 모든 score 불러와서 score2stave 후, padding 준 거 저장
    #         score_path_list = Util.get_all_files(f"{title_path}", EXP[PNG])
    #         for score_path in score_path_list:
    #             Image2Augment.process_image2augment(score_path)

    @staticmethod
    def readimg(img):
        """
        배경이 흰색이든 투명이든 처리하여 Binary 이미지를 생성
        rgb img -> binary img
        input : rgb image
        return : binary image
        """
        # 1. 이미지를 원래의 채널 수와 비트 깊이 그대로 읽음
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        biImg = Image2Augment.img2binary(img)
        return biImg
    
    @staticmethod
    def img2binary(img):
        """
        배경이 흰색이든 투명이든 처리하여 Binary 이미지를 생성
        rgb img -> binary img
        input : rgb image
        return : binary image
        """
        # 1. 이미지를 원래의 채널 수와 비트 깊이 그대로 읽음
        if img is None:
            raise RuntimeError("Image not found or unable to load.")

        # Image has an alpha channel (RGBA)
        if img.shape[-1] == 4:
            # 2. RGB와 알파 채널을 분리
            bgr = img[:, :, :3]
            alpha_channel = img[:, :, 3]

            # 3. RGB 채널이 모두 255인 픽셀(흰색 배경)을 감지하여 해당 알파 채널 값을 0으로 설정
            white_background = np.all(bgr == [255, 255, 255], axis=-1)
            alpha_channel[white_background] = 0

            # 4. 알파 채널을 반전하여 마스크를 생성하고, 이를 그레이스케일 이미지로 사용
            mask = 255 - alpha_channel
            gray = mask
        # Image is RGB
        elif img.shape[-1] == 3:
            # 2. 이미지를 그레이스케일로 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif img.ndim == 2:
            # 3. 이미 그레이스케일
            gray = img

        else:
            raise RuntimeError("Unsupported image type!")

        # cv2.threshold 함수를 사용하여 이진화를 수행
        _, biImg = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # os.makedirs(f"../data/processed-feature/stave/Rock-ver", exist_ok=True)
        # for idx, stave in enumerate([biImg]):
        #     date_time = Util.get_datetime()
        #     cv2.imwrite(
        #         f"../data/processed-feature/stave/Rock-ver/Rock-ver-stave_{idx+1}_{date_time}.png",
        #         stave,
        #     )
        #     print(idx+1, "--shape: ", stave.shape)

        # return 배경이 검정색인 Binary Image
        return biImg

    @staticmethod
    def process_image2augment(args, rgb_path):
        """
        input: rgb image path
        output : augment image
        """
        # augment 적용하려면 2D 여야함.
        biImg = Image2Augment.readimg(rgb_path)
        biImg = 255 - biImg

        # Apply augmentations
        awgn_image = Image2Augment.apply_awgn(biImg, args.augment.awgn_sigma)
        et_small_image = Image2Augment.apply_elastic_transform(
            biImg,
            args.augment.et_small_alpha_training,
            args.augment.et_small_sigma_evaluation,
        )
        et_large_image = Image2Augment.apply_elastic_transform(
            biImg,
            args.augment.et_large_alpha_training,
            args.augment.et_large_sigma_evaluation,
        )
        all_augmentations_image = Image2Augment.apply_all_augmentations(
            biImg,
            args.augment.et_small_alpha_training,
            args.augment.et_small_sigma_evaluation,
            args.augment.et_large_alpha_training,
            args.augment.et_large_sigma_evaluation,
        )

        result = [
            ("origin", biImg),
            ("awgn", awgn_image),
            ("et_small", et_small_image),
            ("et_large", et_large_image),
            ("all_augment", all_augmentations_image),
        ]

        return result

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

        # print("--", distorted_x.shape)
        # print("--", distorted_y.shape)

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
    def save_augment_png(dir_path, image, augment_type):
        """
        save AUGMENT png
        """
        date_time = Util.get_datetime()
        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(
            f"{dir_path}/{augment_type}_{date_time}.png",
            image,
        )
        # print(augment_type, "--AUGMENT shape: ", image.shape)

    @staticmethod
    def resizeimg(args, img):
        # Input: 2D Image
        h, w = img.shape
        # print(">>>>>>>>>>>>>>>>>>>>>>>")

        # 이미지의 가로세로 비율 계산
        ratio = min(args.max_width / w, args.max_height / h)

        # 이미지를 size_w 또는 size_h에 맞춰서 resize
        resized_image = cv2.resize(img, (int(w * ratio), int(h * ratio)))

        # 만약 세로 길이가 size_h를 넘는다면, 다시 size_h에 맞춰서 resize
        if resized_image.shape[0] > args.max_height:
            ratio = args.max_height / resized_image.shape[0]
            resized_image = cv2.resize(
                resized_image, (int(resized_image.shape[1] * ratio), args.max_height)
            )

        img = resized_image

        # 이미지 고정 크기로 맞추기 위해 zero padding
        top_pad = (args.max_height - img.shape[0]) // 2
        bottom_pad = args.max_height - img.shape[0] - top_pad
        left_pad = (args.max_width - img.shape[1]) // 2
        right_pad = args.max_width - img.shape[1] - left_pad

        # 0 0 94 95 상하좌우
        img = np.pad(
            img,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            mode="constant",
            constant_values=255,
        )
        return img
