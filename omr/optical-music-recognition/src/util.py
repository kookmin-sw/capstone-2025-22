from datetime import datetime
import glob
import os

import pandas as pd

from constant.common import CSV, EXP, FEATURE, JSON
from constant.note import CODE2NOTES
from constant.path import DATA_FEATURE_PATH, DATA_RAW_PATH, OSMD


class Util:
    @staticmethod
    def get_datetime():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def get_title(score_path):
        # path 로부터 title 가져옴
        return os.path.basename(os.path.dirname(score_path))

    @staticmethod
    def get_title_from_dir(score_path):
        # "../data/raw/osmd-dataset-v1.0.0/Rock-ver"
        return os.path.basename(score_path)

    @staticmethod
    def get_title_from_featurepath(score_path):
        return os.path.basename(os.path.dirname(os.path.dirname(score_path)))

    @staticmethod
    def get_all_files(parent_folder_path, exp):
        all_file_list = glob.glob(f"{parent_folder_path}/*")
        file_list = [file for file in all_file_list if file.endswith(f".{exp}")]
        return file_list

    @staticmethod
    def get_all_subfolders(folder_path):
        subfolder_paths = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolder_paths.append(item_path)
        return subfolder_paths

    @staticmethod
    def load_feature_from_csv(csv_file_path):
        df = pd.read_csv(csv_file_path)
        print(f"csv shape: {df.shape}")
        return df

    @staticmethod
    def save_feature_csv(title, features, label_type, state):
        """
        state : LABELED_FEATURE | FEATURE
        """
        os.makedirs(f"{DATA_FEATURE_PATH}/{label_type}/{title}/", exist_ok=True)
        date_time = Util.get_datetime()
        save_path = (
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}_{state}_{date_time}.csv"
        )
        df = pd.DataFrame(features)
        df.to_csv(save_path, index=False)
        print(f"{title} - features shape: {df.shape}")

    @staticmethod
    def transform_arr2dict(arr_data):
        print("shape:", arr_data.shape)

        result_dict = {}
        for code, drum in CODE2NOTES.items():
            result_dict[drum] = [row[code] for row in arr_data]
        return result_dict

    @staticmethod
    def get_filepath_from_title(title, exp):
        # title가지고 raw로부터 file path 리턴
        path = f"{DATA_RAW_PATH}/{OSMD}/{title}"
        try:
            file = Util.get_all_files(path, exp)
            return file[0]
        except:
            print(f"!!-- {exp} 파일이 없습니다 : {path} --!!")

    @staticmethod
    def get_csvpath_from_title(title, label_type, state):
        """
        state: FEATURE | LABELED_FEATURE
        """
        # title가지고 processed로부터 feature file path 리턴
        path = f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}_{state}*.{EXP[CSV]}"
        csv_files = glob.glob(path)
        csv_files.sort(reverse=True)

        if csv_files:
            csv_file = csv_files[0]
            return csv_file
        else:
            print(f"!!-- csv 파일이 없습니다 : {path} --!!")
