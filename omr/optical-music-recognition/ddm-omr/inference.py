from configs import getconfig
from sheet2score import SheetToScore
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference single staff image")
    parser.add_argument("filepath", type=str, help="path to staff image")
    parsed_args = parser.parse_args()

    cofigpath = f"ddm-omr/workspace/config.yaml"
    args = getconfig(cofigpath)

    # 예측할 악보
    score_path = parsed_args.filepath

    handler = SheetToScore(args)
    predict_result = handler.predict(score_path)

