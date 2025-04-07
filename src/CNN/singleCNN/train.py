import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio

from model import SimpleCNN, CombinedCNN
from dataset import DrumDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() #모델을 학습 모드로 전환
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader: #입력 데이터와 정답 레이블
        #데이터를 cpu/gpu로 이동
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #매 배치마다 기울기 초기화

        #모델 예측, 손실 계산
        outputs = model(features) #입력 데이터를 모델에 통과시켜 예측값 얻기
        loss = criterion(outputs, labels) #예측과 정답 비교해 오차 계산

        loss.backward() #오차를 바탕으로 가중치(모델 파라미터)를 얼마나 업데이트할지 기울기 계산
        optimizer.step() #계산된 기울기를 이용해 가중치 갱신(실제 학습)

        # 현재 배치의 오차 누적, 정확도 계산
        total_loss += loss.item() * features.size(0)
        _, pred = torch.max(outputs, dim=1) #예측 확률 중 가장 높은 값 반환
        correct += (pred == labels).sum().item() #예측과 정답 비교해 맞은 개수 누적
        total += labels.size(0)
    
    # 한 epoch 동안의 평균 손실과 정확도 
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval() #모델을 평가모드로 전환
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): #학습을 위한 기울기 계산 x
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def singleCNN_train(epochs, device, mel_transform):
    # 데이터셋 & 로더
    train_dataset = DrumDataset('../../../data/train', transform=mel_transform)
    val_dataset   = DrumDataset('../../../data/val', transform=mel_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 모델 생성
    num_classes = len(train_dataset.class_to_idx)
    model = SimpleCNN(num_classes).to(device)
    # 손실함수와 옵티마이저
    criterion = nn.CrossEntropyLoss() #손실함수 : 모델의 예측과 정답 비교해 오차 계산
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #옵티마이저 : 계산된 기울기(gradient)를 이용해 모델의 가중치 업데이트

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    # 모델 저장
    model_path = f"drum_model_simpleCNN_{epochs}.pt"
    torch.save(model.state_dict(), model_path)

    return model_path

def combinedCNN_train(epochs, device, mel_transform):
    # 데이터셋 & 로더
    train_dataset = DrumDataset('../../../data/train', transform=mel_transform)
    val_dataset   = DrumDataset('../../../data/val', transform=mel_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 모델 생성
    num_classes = len(train_dataset.class_to_idx)
    model = CombinedCNN(num_classes).to(device)
    # 손실함수와 옵티마이저
    criterion = nn.CrossEntropyLoss() #손실함수 : 모델의 예측과 정답 비교해 오차 계산
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #옵티마이저 : 계산된 기울기(gradient)를 이용해 모델의 가중치 업데이트

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    # 모델 저장
    model_path = f"drum_model_combinedCNN_{epochs}.pt"
    torch.save(model.state_dict(), model_path)

    return model_path
