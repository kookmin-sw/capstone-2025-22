import torch
import torchaudio
import os, sys

from train import multiCNN_train
from inference import CNN_inference

previous_folder = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(previous_folder)

from utils import get_mel_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU or CPU 선택

torch_mel_transform = get_mel_transform("torch")

# 1.학습
epochs = 30
# model_path = multiCNN_train(epochs, device, torch_mel_transform)

# model_path = "drum_model_multiCNN_30.pt"
model_path = "drum_model_multiCNN_30_8layer_train5000_val1500.pt"
# 2.예측
print(model_path)
CNN_inference(device, torch_mel_transform, model_path, test_data_folder_path="../../test_data")