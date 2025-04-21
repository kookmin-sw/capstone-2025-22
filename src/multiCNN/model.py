import torch.nn as nn
import torch.nn.functional as F

class MultiCNN(nn.Module):
    def __init__(self):
        super(MultiCNN, self).__init__()
        self.class_labels = ["KD", "SD", "CY", "TT", "HH"]
        self.num_classes = len(self.class_labels)
        self.DEBUG_FLAG = False

        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # 배치 정규화 추가
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 두 번째 컨볼루션 블록
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # 배치 정규화 추가
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

                # 세 번째 컨볼루션 블록
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 네 번째 컨볼루션 블록
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # 배치 정규화 추가
            nn.ReLU()
        )

        # 높이와 너비를 고정된 크기로 출력
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # 1D 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.5),  # 드롭아웃 추가 (50%의 뉴런을 무작위로 비활성화)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # 1D 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.3),  # 드롭아웃 추가 (30%의 뉴런을 무작위로 비활성화)
            nn.Linear(128, self.num_classes),
            # nn.Sigmoid() # 시그모이드 활성화 함수 추가 (다중 레이블 분류를 위해) 각 클래스의 확률을 0~1 사이로 독립적으로 계산 #BCEWithLogitsLoss 내부에서 자체적으로 실행
        )
    
    def forward(self, x):
        if self.DEBUG_FLAG : print("Input shape :", x.shape)
        out = self.conv1(x)
        if self.DEBUG_FLAG : print("After conv1:", out.shape)
        out = self.conv2(out)
        if self.DEBUG_FLAG : print("After conv2:", out.shape)
        out = self.conv3(out)
        if self.DEBUG_FLAG : print("After conv3:", out.shape)
        out = self.conv4(out)
        if self.DEBUG_FLAG : print("After conv4:", out.shape)

        out = self.adaptive_pool(out)
        if self.DEBUG_FLAG : print("After adaptive_pool:", out.shape)

        out = out.view(out.size(0), -1) # 평탄화
        if self.DEBUG_FLAG : print("After flatten:", out.shape)

        out = self.classifier(out)

        return out