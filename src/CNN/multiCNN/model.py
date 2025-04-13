import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiCNN, self).__init__()
        self.class_labels = ["KD", "SD", "CY", "TT", "HH"]  # 클래스 레이블 추가
        self.num_classes = len(self.class_labels)

        #Conv2d: 입력(1채널) -> 출력(16채널)
        #MaxPool2d: 높이와 너비 크기를 대략 절반으로 줄이기
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) 
        )

        #Conv2d: 입력(16채널) -> 출력(32채널)
        #MaxPool2d: 높이와 너비 크기를 대략 절반으로 줄이기
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # 높이와 너비를 고정된 크기(16, 16)로 출력
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        # 모델 출력을 1차원 벡터로 변환해서 클래스 최종 분류
        self.classifier = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            
            nn.Sigmoid()  # 시그모이드 활성화 함수 추가 (다중 레이블 분류를 위해) 각 클래스의 확률을 0~1 사이로 독립적으로 계산
        )
    
    def forward(self, x):
        print("Input shape:", x.shape)
        out = self.conv1(x)
        print("After conv1:", out.shape)
        out = self.conv2(out)
        print("After conv2:", out.shape)
        out = self.adaptive_pool(out) # 항상 32x16x16 크기로 출력됨
        print("After adaptive_pool:", out.shape)
        out = out.view(out.size(0), -1) # 고차원 32x16x16 -> 1차원 8192 벡터로 펼침
        print("After flatten:", out.shape)
        out = self.classifier(out)

        return out