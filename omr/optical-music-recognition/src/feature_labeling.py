import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


from constant.common import (
    CURSOR,
    DATASET_DF,
    EXP,
    FEATURE,
    JSON,
    KEY_CURSOR_LIST,
    KEY_MEASURE_LIST,
    LABELED_FEATURE,
    MULTI_LABEL,
    NOTE_PAD,
    PNG,
    STAVE_WIDTH,
    XML,
)
from constant.note import (
    NOTES2CODE,
    NOTES_HEIGHT,
    REST_EIGHTH,
    REST_HALF,
    REST_QUARTER,
    REST_WHOLE,
    TYPE_EIGHTH,
    TYPE_HALF,
    TYPE_QUARTER,
    TYPE_WHOLE,
    REST_16th,
    TYPE_16th,
)
from constant.path import DATA_FEATURE_PATH, DATA_RAW_PATH, OSMD
from util import Util


class FeatureLabeling:
    @staticmethod
    def process_all_feature2label():
        """
        processed-feature/ 의 faeture에 label 더해 새로운 feature-labeled csv 저장
        """
        # processed-feature/multi-label 에서 title들 가져오기
        feature_path_ = f"{DATA_FEATURE_PATH}/{MULTI_LABEL}"
        feature_title_path_list = Util.get_all_subfolders(feature_path_)

        # title 마다 score2stave
        for feature_path in feature_title_path_list:
            title = Util.get_title_from_dir(feature_path)
            # XML, json 파일 경로
            file_parent = f"{DATA_RAW_PATH}/{OSMD}/{title}"

            # 라벨 가져오기
            pitch_list, duration_list = FeatureLabeling.process_xml2label(title)

            # feature 가져오기
            csv_file_path = Util.get_csvpath_from_title(title, MULTI_LABEL, FEATURE)

            # label_feature 없애기
            feature_df = Util.load_feature_from_csv(csv_file_path)

            # score에 pitch, width 그리기
            # json
            # json_path = f"{file_parent}/{title}.{EXP[JSON]}"
            json_path = Util.get_filepath_from_title(title, EXP[JSON])
            score_path = Util.get_all_files(f"{file_parent}", EXP[PNG])

            try:
                score_path = score_path[0]

                if score_path != None:
                    FeatureLabeling.draw_cursor_on_score(
                        title, MULTI_LABEL, score_path, json_path
                    )
                    FeatureLabeling.draw_label_on_cursor(
                        title, MULTI_LABEL, score_path, json_path, pitch_list
                    )

                FeatureLabeling.process_feature2label(
                    title, json_path, feature_df, pitch_list, duration_list
                )
            except:
                print("해당하는 png가 없습니다.!!", score_path)

    @staticmethod
    def process_feature2label(title, json_path, feature_df, pitch_list, duration_list):
        """
        1. 먼저 가로로 label df 생성 후
        2. feature df + label df.T -> new csv
        """
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        # shape: NOTES_HEIGHT x feature concat
        label_df = pd.DataFrame(
            0, index=range(NOTES_HEIGHT), columns=range(feature_df.shape[0])
        )
        # -- stave는 score의 양옆 padding을 자르게 되니, 실제 cursor size와 달라짐. -> 맨 처음 마디의 x 만큼 sliding
        score_leftpad = data[KEY_MEASURE_LIST][0][0]["left"]

        cursor_list = 0
        # cursorList-2d: row 마디 x col 노트
        for i, cursor in enumerate(data[KEY_CURSOR_LIST]):
            print("len: ", len(cursor))

            for _, point in enumerate(cursor):
                # print("row: ", i, ", col: ", j)

                _, left, _, width = FeatureLabeling.get_cursor_data(point)
                left += i * STAVE_WIDTH - score_leftpad
                # print(i, " : ", left)

                # -- 노트 인식된 곳에, xml에서 뽑아온 걸 매핑
                note_code = [0] * NOTES_HEIGHT
                for pitch in pitch_list[cursor_list]:
                    pitch_idx = NOTES2CODE[pitch]
                    note_code[pitch_idx] = 1
                for duration in duration_list[cursor_list]:
                    duration_idx = NOTES2CODE[duration]
                    note_code[duration_idx] = 1

                # -- shape을 넘길 수 있어서
                right_idx = min(left + width, label_df.shape[1])

                tmp_width = right_idx - left - 2 * NOTE_PAD
                note_code_df = [note_code.copy() for _ in range(tmp_width)]
                transpose_data = np.transpose(note_code_df)

                label_df.loc[:, left + NOTE_PAD : left + width - 1 - NOTE_PAD] = (
                    transpose_data
                )
                cursor_list += 1
            print("----------------------------")

        print(
            f"pitch_list len: {len(pitch_list)}, duration_list len: {len(duration_list)}, cursor len: {cursor_list}"
        )

        label_df = np.transpose(label_df)

        # 각 열에 이름 붙이기
        label_df.columns = DATASET_DF["label"]

        merged_df = pd.concat([label_df, feature_df], axis=1)
        Util.save_feature_csv(title, merged_df, MULTI_LABEL, LABELED_FEATURE)

        # print(label_df)

    @staticmethod
    def process_xml2label(title):
        """
        미리 뽑아놓은 feature csv로부터 label을 더해 새로운 feature-labeled csv 저장
        """

        try:
            xml_file_path = Util.get_filepath_from_title(title, EXP[XML])
            pitch_list = FeatureLabeling.extract_pitch(xml_file_path)
            duration_list = FeatureLabeling.extract_duration(xml_file_path)

            # for i, pitches in enumerate(pitch_list, 1):
            #     print(f"Note {i}: {' '.join(pitches)}")
            return pitch_list, duration_list
        except:
            print("해당하는 XML파일이 없습니다...!!", xml_file_path)

    # feature에 label 매핑
    @staticmethod
    def load_xml_data(file_path: str):
        """
        xml data 불러오기
        """
        try:
            tree = ET.parse(file_path)  # XML 파일을 파싱
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            print(f"XML 파일을 파싱하는 동안 오류가 발생했습니다: {e}")
            return None

    @staticmethod
    def extract_pitch(xml_file):
        """
        1. multiple pitch 추출
        <chord/> <-  얘 있으면 동시에 친 거임
        <unpitched>
            <display-step>A</display-step>
            <display-octave>5</display-octave>
        </unpitched>

        2. !!!!!!!!!!!!!예외!!!!!!!!!!!!!
        - grace note 제외

        3. 쉼표 추출
        <note>
            <rest/>
            <duration>48</duration>
            <type>quarter</type>
        </note>

        output : [['G5'], ['G5'], ['G5'], ['C5'], ['C5'], ['F4', 'A5'], ...]
        """

        def extract_step_octave(pitch_element):
            """
            step, octave 추출
            <unpitched>
                <step>C</step>
                <octave>5</octave>
            </unpitched>
            """
            step = pitch_element.find("display-step").text
            octave = pitch_element.find("display-octave").text
            return step, octave

        # XML 파일 파싱
        root = FeatureLabeling.load_xml_data(xml_file)

        pitch_list = []
        chord_list = []

        # 모든 <note> 엘리먼트를 찾습니다.
        for note in root.iter("note"):
            # <grace> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_grace = note.find("grace") is not None
            if is_grace:
                print("grace!")
                continue

            # <rest> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_rest = note.find("rest") is not None
            if is_rest:
                rest_element = note.find("type").text
                if rest_element == TYPE_QUARTER:
                    pitch_list.append([REST_QUARTER])
                elif rest_element == TYPE_EIGHTH:
                    pitch_list.append([REST_EIGHTH])
                elif rest_element == TYPE_HALF:
                    pitch_list.append([REST_HALF])
                elif rest_element == TYPE_WHOLE:
                    pitch_list.append([REST_WHOLE])
                elif rest_element == TYPE_16th:
                    pitch_list.append([REST_16th])
                continue

            pitch_elements = note.findall("./unpitched")
            # <chord> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_chord = note.find("chord") is not None
            # 만약 <chord> 엘리먼트를 가진 <note> 엘리먼트라면, 계속 추가
            if is_chord:
                for pitch_element in pitch_elements:
                    step, octave = extract_step_octave(pitch_element)
                    chord_list.append(step + octave)
            else:
                for pitch_element in pitch_elements:
                    step, octave = extract_step_octave(pitch_element)
                    chord_list = []  # -- 초기화
                    chord_list.append(step + octave)
                    pitch_list.append(chord_list)

        return pitch_list

    @staticmethod
    def extract_duration(xml_file: str):
        """
        1. duration 추출
        !!!!! multilabel 없음. (동시에 여러 개 duration 없음.)

        <chord/> <-  얘 있으면 동시에 친 거임
        <unpitched>
            <display-step>A</display-step>
            <display-octave>5</display-octave>
        </unpitched>

        2. !!!!!!!!!!!!!예외!!!!!!!!!!!!!
        - grace note 제외

        3. 쉼표 추출
        <note>
            <rest/>
            <duration>48</duration>
            <type>quarter</type>
        </note>

        4. 셋잇단음표 예외처리... (임시)
        - 그냥 생긴 걸로 4분 음표로 처리

        output : [["0.500"], ["0.500"], ["1.000"], ["1.000"], ["0.750"], ["0.750"], ...]
        """

        def extract_division(xml_file: str):
            root = FeatureLabeling.load_xml_data(xml_file)
            if root is not None:
                # 'divisions' 요소를 찾아서 값을 가져옴
                divisions_element = root.find(".//divisions")

                if divisions_element is not None:
                    divisions_value = int(divisions_element.text)
                    print("-- Divisions 값:", divisions_value)
                    return divisions_value
                else:
                    print("-- !! no 'divisions' element.")
            else:
                print("-- !! XML parse error.")

        # 셋잇단음표인 경우, 음표의 duration을 계산하는 함수
        def calculate_triplet_duration(duration, actual_notes, normal_notes):
            triplet_duration = (duration * actual_notes) / normal_notes
            return triplet_duration

        # -- duration = duration_value/division
        division = extract_division(xml_file)
        # XML 파일 파싱
        root = FeatureLabeling.load_xml_data(xml_file)

        duration_list = []

        # 모든 <note> 엘리먼트를 찾습니다.
        for note in root.iter("note"):
            # <grace> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_grace = note.find("grace") is not None
            if is_grace:
                print("grace!")
                continue

            # <rest> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_rest = note.find("rest") is not None
            if is_rest:
                rest_element = note.find("type").text
                if rest_element == TYPE_QUARTER:
                    duration_list.append([REST_QUARTER])
                elif rest_element == TYPE_EIGHTH:
                    duration_list.append([REST_EIGHTH])
                elif rest_element == TYPE_HALF:
                    duration_list.append([REST_HALF])
                elif rest_element == TYPE_WHOLE:
                    duration_list.append([REST_WHOLE])
                elif rest_element == TYPE_16th:
                    duration_list.append([REST_16th])
                continue

            # <chord> 엘리먼트를 가진 <note> 엘리먼트인지 확인
            is_chord = note.find("chord") is not None
            # 만약 <chord> 엘리먼트를 가진 <note> 엘리먼트라면, pass
            if is_chord:
                continue

            # 각 note 요소에서 셋잇단음표인 경우 음표의 duration 계산
            duration_value = int(note.find("duration").text)
            time_modification = note.find("time-modification")

            if time_modification is not None:
                actual_notes = int(time_modification.find("actual-notes").text)
                normal_notes = int(time_modification.find("normal-notes").text)
                triplet_duration_value = calculate_triplet_duration(
                    duration_value, actual_notes, normal_notes
                )
                # print("셋잇단음표인 경우 음표의 duration:", triplet_duration)
                duration = float(triplet_duration_value) / float(division)
            else:
                duration = float(duration_value) / float(division)
            duration = f"{duration:.3f}"
            duration_list.append([duration])

        return duration_list

    @staticmethod
    def get_cursor_data(point):
        """
        cursor 확인
        """
        # cursor 정보는 1024 기준이라서 x2
        top = int(point["top"])
        left = int(point["left"])
        height = int(point["height"])
        width = int(point["width"])

        return top, left, height, width

    @staticmethod
    def draw_cursor_on_score(title, label_type, image_path, json_path):
        """
        OSMD로 추출한 cursor 위치값을 score에 그려보기
        """
        # 이미지 읽어오기
        image = cv2.imread(image_path)

        # JSON 파일 읽어오기
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        # 빨간색 네모 그리기
        for cursor in data["cursorList"]:
            for point in cursor:
                top, left, height, width = FeatureLabeling.get_cursor_data(point)
                cv2.rectangle(
                    image,
                    (left + NOTE_PAD, top),
                    (left + width - NOTE_PAD, top + height),
                    (0, 0, 255),
                    2,
                )
        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}-{CURSOR}-{date_time}.{EXP[PNG]}",
            image,
        )

    @staticmethod
    def draw_label_on_cursor(title, label_type, image_path, json_path, pitch_list):
        """
        OSMD로 추출한 label을 cursor에 그려보기
        """
        # 이미지 읽어오기
        image = cv2.imread(image_path)

        # JSON 파일 읽어오기
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        print("===========================")
        print("data[cursorList] : ", len(data[KEY_CURSOR_LIST]))
        print("pitch_list: ", len(pitch_list))

        # -- data[KEY_CURSOR_LIST
        # ] 는 2D로 row: stave, col: cursor
        # -- pitch_list 는 1D로 cursor 쭉 나열되어 있음.

        cursor_idx = 0
        # cursorList-2d: row 마디 x col 노트
        for _, cursor in enumerate(data[KEY_CURSOR_LIST]):
            for _, point in enumerate(cursor):
                top, left, _, _ = FeatureLabeling.get_cursor_data(point)

                # print("!!----------- pitch_list -------------!!", cursor_idx)
                # print(len(pitch_list))

                # print(pitch_list[cursor_idx])
                # -- 노트 인식된 곳에, xml에서 뽑아온 걸 매핑
                # if len(pitch_list) <= cursor_idx:
                #     print("으엥: ", len(pitch_list), cursor_idx)
                #     continue
                for idx, pitch in enumerate(pitch_list[cursor_idx]):
                    cv2.putText(
                        image,
                        pitch,
                        (left + NOTE_PAD, top + NOTE_PAD + idx * 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cursor_idx += 1

        date_time = Util.get_datetime()
        cv2.imwrite(
            f"{DATA_FEATURE_PATH}/{label_type}/{title}/{title}-label-{date_time}.png",
            image,
        )

    @staticmethod
    def load_all_labeled_feature_file():
        """
        1. 모든 processed-feature에 있는 labeled-feature.csv 파일들 가져오기
        2. csv 파일들 이어 붙이기
        """
        # osmd title paths 가져오기
        title_path_ = f"{DATA_FEATURE_PATH}/{MULTI_LABEL}"
        title_path_list = Util.get_all_subfolders(title_path_)

        labeled_feature_file_list = []
        for title_path in title_path_list:
            # files = Util.get_all_files(f"{title_path}", EXP[CSV])
            title = Util.get_title_from_dir(title_path)
            file = Util.get_csvpath_from_title(title, MULTI_LABEL, LABELED_FEATURE)
            labeled_feature_file_list.append(file)
            # for file in files:
            #     if LABELED_FEATURE in file:

        # dataframe으로 합치기
        combined_df = pd.DataFrame()
        for labeled_feature in labeled_feature_file_list:
            feature_file = pd.read_csv(labeled_feature)
            combined_df = pd.concat([combined_df, feature_file], ignore_index=True)
            del feature_file

        print(
            "-- ! 로딩 완료 ! --",
            "data shape:",
            combined_df.shape,
        )
        print("-- ! features ! -- ")
        print(combined_df)

        return combined_df
