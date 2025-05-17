import torch.nn as nn
import torch.nn.functional as F

class MultiCNN(nn.Module):
    def __init__(self):
        super(MultiCNN, self).__init__()
        self.class_labels = ["KD", "SD", "CY", "TT", "HH"]
        self.num_classes = len(self.class_labels)
        self.DEBUG_FLAG = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # 1D 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.5),  # 드롭아웃 추가 (50%의 뉴런을 무작위로 비활성화)

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # 1D 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.4),  # 드롭아웃 추가 (40%의 뉴런을 무작위로 비활성화)

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, self.num_classes),
            # nn.Sigmoid() # 시그모이드 활성화 함수 추가 (다중 레이블 분류를 위해) 각 클래스의 확률을 0~1 사이로 독립적으로 계산 #BCEWithLogitsLoss 내부에서 자체적으로 실행
        )

    def forward(self, x):
        if self.DEBUG_FLAG: print("Input:", x.shape)
        x = self.conv1(x)
        if self.DEBUG_FLAG: print("conv1:", x.shape)
        x = self.conv2(x)
        if self.DEBUG_FLAG: print("conv2:", x.shape)
        x = self.conv3(x)
        if self.DEBUG_FLAG: print("conv3:", x.shape)
        x = self.conv4(x)
        if self.DEBUG_FLAG: print("conv4:", x.shape)

        x = self.adaptive_pool(x)
        if self.DEBUG_FLAG: print("pooled:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        if self.DEBUG_FLAG: print("flattened:", x.shape)

        out = self.classifier(x)
        return out