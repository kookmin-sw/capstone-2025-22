import os
import xml.etree.ElementTree as ET

from constant.common import ANNOTATION, EXP, TRANSFORMER, TXT
from constant.path import DATA_FEATURE_PATH
from util import Util


class Xml2Annotation:
    # 각 measure에 대한 처리 함수
    @staticmethod
    def process_measure(measure, division):
        stave_string = ""

        for element in measure:
            if element.tag == "attributes":
                # # clef 처리
                # clef = element.find('clef')
                # if clef is not None:
                #     stave_string += f"clef-{clef.find('sign').text}"

                # time signature 처리
                time = element.find("time")
                if time is not None:
                    stave_string += f"timeSignature-{time.find('beats').text}/{time.find('beat-type').text}+"

            elif element.tag == "note":
                if element.find("rest") is not None:
                    stave_string += "rest"
                elif element.find("unpitched") is not None:
                    # note 정보 처리
                    pitch = element.find("unpitched")
                    stave_string += f"note-{pitch.find('display-step').text}{pitch.find('display-octave').text}"

                rhythm = element.find("type").text
                stave_string += f"_{rhythm}+"

                if element.find("grace") is None:
                    # 0.25 0.375 0.5 0.75 1.0 1.5 2.0 3.0 4.0 중에서
                    # 0.375 0.75 1.5 3 이면 뒤에 . 붙이기
                    duration_value = element.find("duration").text
                    duration = float(duration_value) / float(division)
                    d_ = "." if duration in [0.375, 0.75, 1.5, 3] else ""
                    stave_string = stave_string[:-1] + f"{d_}+"

                # chord 여부에 따라 | 또는 + 추가
                if element.find("chord") is not None:
                    # clef-percussion+note-F4_quarter+(<- 이걸 '|'로) note-A5_quarter+
                    # note-G5_eighth+note-C5_eighth+note-G5_eighth+ 에서 끝의 + 빼고 reverse 한 후
                    # replace('+', '|', 1) 1개의 +만 |로 replace 한 후
                    # 다시 reverse 후, 뒤에 + 붙여서 완성!
                    components = stave_string[:-1][::-1]
                    components = components.replace("+", "|", 1)
                    stave_string = components[::-1] + "+"

        return stave_string

    @staticmethod
    def process_xml2annotation(xml_path):
        # 0. stave 마다 새로운 string 생성: print new-system 일 때마다
        # 1. clef-purcussion 삽입: xml에선 맨 처음에만 나오니까 매번 삽입
        # 2. time-signature 있으면 삽입
        # 3. note 삽입: pitch, duration
        # rest 삽입!!
        # 4. if 동시에 나온 note일 시, | 으로 구분
        # 5. else + 로 연결

        # MusicXML 파일을 파싱하여 ElementTree 객체 생성
        tree = ET.parse(xml_path)
        root = tree.getroot()
        divisions_element = root.find(".//divisions")
        if divisions_element is not None:
            divisions_value = int(divisions_element.text)
        division = divisions_value

        # 각 stave에 대한 문자열을 저장할 리스트
        stave_strings = []

        # measure 태그를 가진 모든 element에 대해 처리
        stave_tmp = "clef-percussion+"
        for measure in root.findall(".//measure"):
            # print 태그 확인하여 new-system이 있는 경우에는 새로운 문자열로 시작
            if (
                measure.find("print") is not None
                and measure.find("print").get("new-system") == "yes"
            ):
                stave_strings.append(stave_tmp[:-1])
                stave_tmp = "clef-percussion+"

            stave_tmp += Xml2Annotation.process_measure(measure, division) + "barline+"

        print(stave_strings)

        title = Util.get_title(xml_path)
        Xml2Annotation.save_strings_to_txt(stave_strings, TRANSFORMER, title)

    @staticmethod
    def save_strings_to_txt(string_list, label_type, title):
        folder_path = f"{DATA_FEATURE_PATH}/{label_type}/{title}/{ANNOTATION}"
        os.makedirs(folder_path, exist_ok=True)

        for i, string in enumerate(string_list):
            date_time = Util.get_datetime()
            with open(
                f"{folder_path}/{title}_{ANNOTATION}_{i+1}_{date_time}.txt", "w"
            ) as file:
                file.write(string)
