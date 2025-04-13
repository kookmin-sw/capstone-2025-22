import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio

from model import MultiCNN
from dataset import DrumDataset

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    for inputs, labels in train_loader: #입력 데이터와 정답 레이블
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 순전파
        optimizer.zero_grad() #매 배치마다 기울기 초기화
        outputs = model(inputs)
        
        # 손실 계산
        loss = criterion(outputs, labels) #예측과 정답 비교해 오차 계산
        
        # 역전파
        loss.backward() #오차를 바탕으로 가중치(모델 파라미터)를 얼마나 업데이트할지 기울기 계산
        optimizer.step() #계산된 기울기를 이용해 가중치 갱신(실제 학습)
        
        # 손실 누적
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # 다중 레이블 예측 정확도 계산
        predictions = (outputs > 0.5).float()  # 임계값 0.5로 이진 예측
        correct_predictions += (predictions == labels).float().sum().item()
        
    # 전체 에폭의 손실과 정확도 계산
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / (total_samples * len(model.class_labels))  # 전체 예측 수로 나눔
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval() #모델을 평가모드로 전환
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad(): #학습을 위한 기울기 계산 x
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 다중 레이블 예측 정확도 계산
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).float().sum().item()
    
    val_loss = running_loss / total_samples
    val_acc = correct_predictions / (total_samples * len(model.class_labels))
    
    return val_loss, val_acc

def multiCNN_train(epochs, device, mel_transform):
    # 데이터셋 & 로더
    train_dataset = DrumDataset('../../../data/train', transform=mel_transform)
    val_dataset   = DrumDataset('../../../data/val', transform=mel_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 모델 생성
    num_classes = len(train_dataset.class_labels)
    model = MultiCNN(num_classes).to(device)
    # 손실함수와 옵티마이저
    criterion = nn.BCELoss()  #손실함수 : 모델의 예측과 정답 비교해 오차 계산. 다중 레이블 분류를 위해 Binary Cross Entropy Loss 사용
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #옵티마이저 : 계산된 기울기(gradient)를 이용해 모델의 가중치 업데이트

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    # 모델 저장
    model_path = f"drum_model_multiCNN_{epochs}.pt"
    torch.save(model.state_dict(), model_path)

    return model_path