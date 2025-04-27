import os
from process_data.image2augment import Image2Augment
from configs import getconfig
from produce_data.data_processing import DataProcessing

import sys

# sys.path.append("/mnt/c/Users/wotjr/Documents/Github/optical-music-recognition/ddm-omr")
from staff2score import StaffToScore
from sheet2score import SheetToScore

cofigpath = f"workspace/config.yaml"

args = getconfig(cofigpath)
# DataProcessing.process_all_score2measure(args, True)

staff2score = StaffToScore(args)
staff2score.training()
# staff2score.test()

# x_raw_path_list = [
#     f"../data/test/Rock-ver_measure_02_2024-05-19_05-31-40.png",
#     f"../data/test/test_img.png",
# ]
# x_preprocessed_list = []
# for x_raw_path in x_raw_path_list:
#     biImg = Image2Augment.readimg(x_raw_path)
#     biImg = 255 - biImg
#     x_preprocessed_list.append(Image2Augment.resizeimg(args, biImg))

# print("전처리 후 x 개수: ", len(x_preprocessed_list))


# staff2score.model_predict(x_preprocessed_list)

# # sheet2stave = SheetToScore(args)
# # sheet2stave.sheet2stave()
