import json
import os
import cv2
from util import Util


class Score2Measure:
    @staticmethod
    def extract_measure_from_score(args, img, measure_pos_data):
        """
        score에서 measure 추출
        주의!! height는 정말 5선지 높이이기 때문에, 일정 offset만큼 늘려서 잘라줘야 함
        """

        """[...,
            [
            n번째 measure의 1번째 마디
            {
                "top": 2930,
                "left": 50,
                "height": 40,
                "width": 274,
                "timestamp": 91.20833333333333
            },
            n번째 measure의 2번째 마디
            {
                "top": 2930,
                "left": 324.492,
                "height": 40,
                "width": 196,
                "timestamp": 92.125
            }
            ],
            ...]
        """
        measure_list = []
        measure_stats_list = []
        staff_pos_list = measure_pos_data[args.jsonkeys.measure_list]

        PAD = 2
        for measure_pos_list in staff_pos_list:
            for measure_pos in measure_pos_list:
                # 맨 처음 마디와 끝 마디의 길이만큼 자르기 - 마디 한 개만 있는 경우도 상관 없음.
                # start_pos = measure_pos[0]
                # end_pos = measure_pos[-1]

                x = round(measure_pos["left"] + PAD)

                t_y = round((measure_pos["top"] - args.measure_height / 2))
                y = max(t_y, 0)

                h = args.measure_height

                w = round(measure_pos["width"] + PAD)

                measure = img[y : y + h, x : x + w]
                measure_list.append(measure)
                measure_stats_list.append([x, y, w, h])

        return measure_list, measure_stats_list

    @staticmethod
    def transform_score2measure(args, title, score_path, is_draw_save=False):
        """
        score로부터 measure image추출
        """
        img = cv2.imread(score_path)

        (h, w) = img.shape[:2]
        target_width = args.score_width

        # 비율 계산
        ratio = target_width / float(w)
        target_height = int(h * ratio)

        # 이미지 리사이즈
        img = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        # cv2.imwrite(
        #     f"output.png",
        #     img,
        # )

        # JSON 파일 읽어오기
        json_path = Util.get_filepath_from_title(args, title, "json")
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        measure_list, measure_stats_list = Score2Measure.extract_measure_from_score(
            args, img, data
        )

        # (확인용) 악보 위에 measure 그리기
        if is_draw_save:
            Score2Measure.draw_measure_on_score(
                args, title, score_path, measure_stats_list
            )

        return measure_list, measure_stats_list

    @staticmethod
    def save_png(path, img):
        """
        save img
        """
        cv2.imwrite(f"{path}.png", img)
        print("-- shape: ", img.shape)

    @staticmethod
    def draw_measure_on_score(args, title, score_path, measure_stats_list):
        print(measure_stats_list)
        """
        OSMD로 추출한 cursor 위치값을 score에 그려보기
        """
        # 이미지 읽어오기
        image = cv2.imread(score_path)

        for stave_stats in measure_stats_list:
            x, y, w, h = stave_stats
            cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 1)  # 사각형 그리기

        dir_path = f"{args.filepaths.feature_path.seq}/{title}"
        os.makedirs(f"{dir_path}", exist_ok=True)
        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{dir_path}/{title}_{args.measure}_drawcursor_{date_time}.png",
            image,
        )
