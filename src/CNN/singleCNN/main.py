from dataset import DrumDataset
import torch
import torchaudio

from train import singleCNN_train, combinedCNN_train
from inference import CNN_inference
from utils import get_mel_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU or CPU 선택

torch_mel_transform = get_mel_transform("torch")

#학습
epochs = 30
# model_path = combinedCNN_train(epochs, device, torch_mel_transform)

model_path = "drum_model_combinedCNN_30.pt"
#예측
print(model_path)
CNN_inference(device, torch_mel_transform, model_path, test_data_folder_path="../../test_data")