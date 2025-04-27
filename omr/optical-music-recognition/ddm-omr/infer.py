from configs import getconfig
from sheet2score import SheetToScore

def inference(score): 
    cofigpath = f"/home/code/optical-music-recognition/ddm-omr/workspace/config_server.yaml"
    args = getconfig(cofigpath)

    handler = SheetToScore(args)
    predict_result = handler.inferSheetToXml(score)
    return predict_result