import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCNN(nn.Module):
    def __init__(self, num_classes = 4):
        super(MultiCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) 
        )
        #Conv2d: 입력(1채널) -> 출력(16채널)
        #MaxPool2d: 높이와 너비 크기를 대략 절반으로 줄이기
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        #Conv2d: 입력(16채널) -> 출력(32채널)
        #MaxPool2d: 높이와 너비 크기를 대략 절반으로 줄이기

        # 높이와 너비를 고정된 크기(16, 16)로 출력
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        # 모델 출력을 1차원 벡터로 변환해서 클래스 최종 분류
        self.classifier = nn.Sequential(
            # nn.Linear(32 * 16 * 16, 128),
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # print("Input shape:", x.shape)
        out = self.conv1(x)
        # print("After conv1:", out.shape)
        out = self.conv2(out)
        # print("After conv2:", out.shape)
        out = self.adaptive_pool(out) # 항상 32x16x16 크기로 출력됨
        # print("After adaptive_pool:", out.shape)
        out = out.view(out.size(0), -1) # 고차원 32x16x16 -> 1차원 8192 벡터로 펼침
        # print("After flatten:", out.shape)
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