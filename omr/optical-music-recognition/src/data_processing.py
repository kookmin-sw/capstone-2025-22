import numpy as np
import pandas as pd
from constant.common import (
    AUGMENT,
    AUGMENT_SET,
    DATASET_DF,
    EXP,
    FEATURE,
    MULTI_LABEL,
    PAD_STAVE,
    PNG,
)
from constant.path import DATA_FEATURE_PATH
from score2stave import Score2Stave
from util import Util


class DataProcessing:
    @staticmethod
    def process_all_score2stave():
        """
        processed-feature/multi-label/.../augment 에 있는 모든 score 를 padding stave로 변환해서 img(선택), csv로 저장
        """
        # osmd title paths 가져오기
        title_path_ = f"{DATA_FEATURE_PATH}/{MULTI_LABEL}"
        title_path_list = Util.get_all_subfolders(title_path_)
        # title 마다 5장 augment img를 score2stave
        for title_path in title_path_list:
            # 모든 score 불러와서 score2stave 후, padding 준 거 저장
            # -- score -> stave
            title = Util.get_title_from_dir(title_path)
            score_path_list = Util.get_all_files(f"{title_path}/{AUGMENT}", EXP[PNG])
            merged_df = pd.DataFrame()
            for score_path in score_path_list:
                # image에 들어간 augmentation type으로 분류
                # orign | awgn | et_small | et_large | all_augmentations
                augment_set = AUGMENT_SET.keys()

                for aug in augment_set:
                    if aug in score_path:
                        stave_data = DataProcessing.process_score2stave(
                            title, aug, score_path
                        )
                        stave_pd = pd.DataFrame(stave_data)
                        stave_pd.columns = DATASET_DF[aug]

                        merged_df = pd.concat([merged_df, stave_pd], axis=1)

            Util.save_feature_csv(title, merged_df, MULTI_LABEL, FEATURE)

    # feature 생성
    @staticmethod
    def process_score2stave(title, state, score_path):
        stave_list, stave_stats_list = Score2Stave.transform_score2stave(score_path)

        # 악보 위에 stave 그리기
        Score2Stave.draw_stave_on_score(
            title, state, MULTI_LABEL, score_path, stave_stats_list
        )
        # -- padding stave img
        pad_stave_list = Score2Stave.transform_stave2padStave(stave_list)

        # -- 1. image file로부터 feature 저장하려면 아래 코드
        # img_list = get_all_files(f"{DATA_FEATURE_PATH}/{title}", "png")
        # feature_list = transform_staveImg2feature(img_list)
        # -- 2. 위 feature 그대로 쓰려면 아래 코드
        feature_list = pad_stave_list

        # 데이터 이어붙이기
        merged_data = np.concatenate(feature_list, axis=1)

        # 전치
        transposed_data = np.transpose(merged_data)

        # -- save stave img, feature
        Score2Stave.save_stave_png(
            title, pad_stave_list, MULTI_LABEL, f"{PAD_STAVE}-{state}"
        )
        return transposed_data
