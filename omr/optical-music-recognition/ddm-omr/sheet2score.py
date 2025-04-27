import os
import re
import cv2
from produce_data.annotation2xml import Annotation2Xml
from util import Util
from staff2score import StaffToScore
from process_data.image2augment import Image2Augment


class SheetToScore(object):
    def __init__(self, args):
        self.args = args
        self.staff2score = StaffToScore(args)

    def extract_segment_from_score(self, biImg):
        """
        score에서 각 segment 추출
        객체 정보를 함께 반환하는 레이블링 함수
        cnt : 객체 수 + 1 (배경 포함)
        labels : 객체에 번호가 지정된 레이블 맵
        stats : N x 5, N은 객체 수 + 1이며 각각의 행은 번호가 지정된 객체를 의미, 5열에는 x, y, width, height, area 순으로 정보가 담겨 있습니다. x,y 는 좌측 상단 좌표를 의미하며 area는 면적, 픽셀의 수를 의미합니다.
        centroids : N x 2, 2열에는 x,y 무게 중심 좌표가 입력되어 있습니다. 무게 중심 좌표는 픽셀의 x 좌표를 다 더해서 갯수로 나눈 값입니다. y좌표도 동일합니다.
        """
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(biImg)
        return cnt, labels, stats, centroids

    def extract_stave_from_score(self, biImg, cnt, stats):
        """
        stave 추출
        악보
        """
        score_h, score_w = biImg.shape
        PAD = 10

        stave_list = []
        # -- idx 0은 배경이라 제외
        # 1. 임의 widht 이상일 때 stave라고 판단
        for i in range(1, cnt):
            x_s, y_s, w_s, h_s, _ = stats[i]

            x = round(x_s - PAD)

            t_y = round((y_s - PAD))
            y = max(t_y, 0)

            h = h_s + 2 * PAD

            w = round(w_s + 2 * PAD)

            # -- stave 인식
            # -- stave width가 score width와 같지 않은 경우가 있을 수도 있음
            if w >= score_w * 0.3:
                stave = biImg[y : y + h, x : x + w]
                stave_list.append(stave)

        result_stave_list = []
        min_h = score_h / len(stave_list) / 2
        # 1. stave라고 판단된 것 중에 임의 height 이상일 때 stave라고 판단
        for stave in stave_list:
            h, _ = stave.shape
            if h >= 10:
                result_stave_list.append(stave)
        return result_stave_list

    def save_stave(self, title, stave_list):
        """
        save stave list
        """
        os.makedirs(
            f"{self.args.filepaths.feature_path.base}/stave/{title}", exist_ok=True
        )
        for idx, stave in enumerate(stave_list):
            date_time = Util.get_datetime()
            cv2.imwrite(
                f"{self.args.filepaths.feature_path.base}/stave/{title}/{title}-stave_{idx+1}_{date_time}.png",
                stave,
            )
            print(idx + 1, "--shape: ", stave.shape)

    def transform_score2stave(self, score_path):
        """
        score로부터 stave image추출
        """

        biImg = Image2Augment.readimg(score_path)

        (h, w) = biImg.shape[:2]
        target_width = self.args.score_width

        # 비율 계산
        ratio = target_width / float(w)
        target_height = int(h * ratio)

        # 이미지 리사이즈
        biImg = cv2.resize(
            biImg, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        # 배경이 검정색인 binary image일 때 잘 추출하더라
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list

    def transform_scoreImg2stave(self, score):
        """
        score로부터 stave image추출
        """

        biImg = Image2Augment.img2binary(score)

        (h, w) = biImg.shape[:2]
        target_width = self.args.score_width

        # 비율 계산
        ratio = target_width / float(w)
        target_height = int(h * ratio)

        # 이미지 리사이즈
        biImg = cv2.resize(
            biImg, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        # 배경이 검정색인 binary image일 때 잘 추출하더라
        cnt, _, stats, _ = self.extract_segment_from_score(biImg)
        stave_list = self.extract_stave_from_score(biImg, cnt, stats)
        return stave_list

    def stave2measure(self, stave):
        # 냅다 일정 width로 나누기엔 크기 차이가 나니까 담겨있는 정보 차이도 날 거임.
        # stave를 일정 크기로 resize하기 -> height을 맞추기
        h, w = stave.shape
        max_h = self.args.max_height
        max_w = self.args.max_width

        # 이미지의 가로세로 비율 계산
        new_width = int((max_h / h) * w)
        resized_stave = cv2.resize(stave, (new_width, max_h))

        result = []
        start_x = 0  # 현재 이미지의 x 시작점
        _, r_w = resized_stave.shape

        # 이미지 자르기 및 패딩
        while start_x < r_w:
            end_x = min(start_x + max_w, r_w)
            cropped_image = resized_stave[:, start_x:end_x]

            # 남은 부분이 120 픽셀보다 작으면 패딩을 추가합니다.
            if end_x - start_x < max_w:
                padding_needed = max_w - (end_x - start_x)
                # 오른쪽에 패딩을 추가합니다.
                cropped_image = cv2.copyMakeBorder(
                    cropped_image,
                    0,
                    0,
                    0,
                    padding_needed,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            result.append(255 - cropped_image)

            start_x += max_w

        return result

    def preprocessing(self, score_path):
        # ------------ 전처리 ------------------
        stave_list = self.transform_score2stave(score_path)  # stave 추출

        measure_list = []
        for idx, stave in enumerate(stave_list):
            measures = self.stave2measure(stave)  # measure 추출
            measure_list += measures
        x_preprocessed_list = []

        print("measure_list>>>>>>>>>", measure_list)
        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))

        return x_preprocessed_list

    def imagePreprocessing(self, score):
        # ------------ 전처리 ------------------
        stave_list = self.transform_scoreImg2stave(score)  # stave 추출

        measure_list = []
        for idx, stave in enumerate(stave_list):
            measures = self.stave2measure(stave)  # measure 추출
            measure_list += measures
        x_preprocessed_list = []

        for biImg in measure_list:
            x_preprocessed_list.append(Image2Augment.resizeimg(self.args, biImg))

        return x_preprocessed_list

    def postprocessing(self, predict_result):
        # 함수 정의
        def process_string(s):
            # 먼저, | 문자 사이의 공백을 제거
            s = re.sub(r"\s*\|\s*", "|", s)
            # 그 외의 공백은 +로 대체
            s = re.sub(r"\s+", "+", s)
            if s[-1] == "+":
                s = s[:-1]
            return s

        result_list = []
        for res in predict_result:
            result_list.append(process_string(res))
        result = "+".join(result_list)
        print(">>>>", result)
        return result

    def predict(self, score_path):
        x_preprocessed_list = self.preprocessing(score_path)

        # self.save_stave("demo-test", x_preprocessed_list)

        result = self.staff2score.model_predict(x_preprocessed_list)
        postresult = self.postprocessing(result)
        # print(postresult)
        return postresult

    def inferSheetToXml(self, score):
        x_preprocessed_list = self.imagePreprocessing(score)

        # self.save_stave("demo-test", x_preprocessed_list)

        result = self.staff2score.model_predict(x_preprocessed_list)
        postresult = self.postprocessing(result)
        xml_tree = Annotation2Xml.annotation_to_musicxml(postresult)

        return xml_tree
