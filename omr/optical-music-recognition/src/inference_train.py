import os

import argparse
from random import randrange

import cv2
import numpy as np
import pandas as pd
import torch

from configs import getconfig
from constant.common import (
    AUGMENT_SET,
    CHUNK_TIME_LENGTH,
    DATASET_DF,
    MULTI_LABEL,
    STAVE_HEIGHT,
)
from constant.note import NOTES_HEIGHT
from constant.path import DATA_TEST_PATH
from feature_labeling import FeatureLabeling
from staff2score import StaffToScore


if __name__ == "__main__":
    # --
    # parser = argparse.ArgumentParser(description="Inference single staff image")
    # parser.add_argument("filepath", type=str, help="path to staff image")
    # parsed_args = parser.parse_args()

    # os.path.dirname(__file__): 현재 실행 중인 스크립트의 파일 경로
    cofigpath = os.path.join(os.path.dirname(__file__), "workspace", "config.yaml")
    args = getconfig(cofigpath)

    handler = StaffToScore(args)

    imgpath = f"{DATA_TEST_PATH}/test-01.png"

    def convert_img(imgpath):
        imgs = []
        if os.path.isdir(imgpath):
            for item in os.listdir(imgpath):
                imgs.append(handler.readimg(os.path.join(imgpath, item)))
        else:
            imgs.append(handler.readimg(imgpath))
        imgs = torch.cat(imgs).float().unsqueeze(1)

        return imgs

    """
    1 -- resize 전
    (298, 2404, 4)
    2 -- resize 후
    torch.Size([1, 128, 1024])
    torch.float32
    rgbimgs : torch.Size([1, 1, 128, 1024])
    """

    rgbimgs = convert_img(imgpath)
    print(f"rgbimgs : {rgbimgs.shape}")

    # imgs = handler.preprocessing(rgbimgs)
    # print(f"imgs : {imgs.shape}")

    # x : torch.Size([8, 3, 224, 224])
    # patches : torch.Size([8, 196, 768])
    # 배치사이즈 8, 채널 3, h, w = (224, 224)를 갖는 랜덤텐서를 사용하여 텐서연산의 과정을 살펴보도록 하겠습니다.
    # 먼저 BATCHxCxH×W 형태를 가진 이미지를 BATCHxNx(P*P*C)의 벡터로 임베딩을 해주어야 합니다.
    # P는 패치사이즈이며 N은 패치의 개수(H*W / (P*P))입니다.

    # BATCHxCxH×W 형태를 가진 이미지

    

    # handler = StaffToScore(args)
    # predrhythms, predpitchs, predlifts = handler.predict(parsed_args.filepath)

    # mergeds = []
    # for i in range(len(predrhythms)):
    #     predlift = predlifts[i]
    #     predpitch = predpitchs[i]
    #     predrhythm = predrhythms[i]

    #     merge = predrhythm[0] + "+"
    #     for j in range(1, len(predrhythm)):
    #         if predrhythm[j] == "|":
    #             merge = merge[:-1] + predrhythm[j]
    #         elif "note" in predrhythm[j]:
    #             lift = ""
    #             if predlift[j] in (
    #                 "lift_##",
    #                 "lift_#",
    #                 "lift_bb",
    #                 "lift_b",
    #                 "lift_N",
    #             ):
    #                 lift = predlift[j].split("_")[-1]
    #             merge += (
    #                 predpitch[j] + lift + "_" + predrhythm[j].split("note-")[-1] + "+"
    #             )
    #         else:
    #             merge += predrhythm[j] + "+"
    #     mergeds.append(merge[:-1])
    # print(mergeds)
