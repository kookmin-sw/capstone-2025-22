import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 4):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  #입력 채널 1(MelSpectrogram 모노)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 특정 범위에서 가장 큰 값을 추출함으로써 정보를 압축 및 주요 특징만 남김
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  #입력 채널 16(첫 번째 합성곱 출력)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # 적응형 풀링 추가 (출력 크기를 16x16으로 고정)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        

        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.adaptive_pool(out)  # 항상 32x16x16 크기로 출력됨
        out = out.view(out.size(0), -1) #모델 출력을 1차원 벡터로 변환
        out = self.classifier(out)

        return out

class CombinedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CombinedCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 65채널(64 mel bins + 1 waveform)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # print(f"1. 입력 텐서 형태: {x.shape}")
        
        out = self.conv1(x)
        # print(f"2. Conv1 후 형태: {out.shape}")
        
        out = self.conv2(out)
        # print(f"3. Conv2 후 형태: {out.shape}")
        
        out = self.adaptive_pool(out)
        # print(f"4. AdaptivePool 후 형태: {out.shape}, 기대값: [batch, 32, 16, 16]")
        
        out = out.view(out.size(0), -1)
        # print(f"5. 평탄화 후 형태: {out.shape}, 기대값: [batch, 8192]")
        
        out = self.classifier(out)
        return out