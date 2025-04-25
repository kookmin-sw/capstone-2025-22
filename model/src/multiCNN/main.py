import torch

from inference import CNN_inference
import utils

torch_mel_transform = utils.get_mel_transform("torch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU or CPU 선택

model_path = "drum_model_multiCNN_20.pt"
CNN_inference(device, torch_mel_transform, model_path)