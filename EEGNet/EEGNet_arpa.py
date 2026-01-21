from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, accuracy_score
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import pickle
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # 서버 환경에서 GUI 창 띄우지 않음 (멈춤 방지)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from utils import COMBDataset, calculate_metrics, get_patch_time_ms


class EEGNet(nn.Module):
    def __init__(self, n_channels=53, n_timepoints=448, n_classes=6):
        super(EEGNet, self).__init__()
        self.T = n_timepoints
        self.C = n_channels
        self.n_classes = n_classes
        
        # [Layer 1] Spatial Conv (공간 필터)
        # n_channels개 채널의 정보를 하나로 압축
        # Input: (Batch, 1, T, C) -> Output: (Batch, 16, T, 1)
        self.conv1 = nn.Conv2d(1, 16, (1, n_channels), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # [Layer 2] Temporal Conv (시간 필터)
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1)) 
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # [Layer 3] Depthwise/Separable Conv
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer 차원 계산
        self._calculate_fc_input_size(n_channels, n_timepoints)
        
        # FC Layer - n_classes 출력 (5-class classification)
        self.fc1 = nn.Linear(self.fc_input_size, n_classes)
        
    def _calculate_fc_input_size(self, n_channels, n_timepoints):
        """Forward pass를 통해 FC layer 입력 크기 계산"""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_timepoints, n_channels)
            x = F.elu(nn.Conv2d(1, 16, (1, n_channels), padding=0)(x))
            x = x.permute(0, 3, 1, 2)
            x = nn.ZeroPad2d((16, 17, 0, 1))(x)
            x = F.elu(nn.Conv2d(1, 4, (2, 32))(x))
            x = nn.MaxPool2d(2, 4)(x)
            x = nn.ZeroPad2d((2, 1, 4, 3))(x)
            x = F.elu(nn.Conv2d(4, 4, (8, 4))(x))
            x = nn.MaxPool2d((2, 4))(x)
            self.fc_input_size = x.numel()

    def forward(self, x):
        # Layer 1: Spatial Learning
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2: Temporal Learning
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.pooling2(x)
        
        # Layer 3: High-level Feature Learning
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)  # CrossEntropyLoss에서 softmax 처리
        return x


def evaluate(model, data_loader, params=["acc"]):
    """Multi-class classification용 evaluate 함수"""
    model.eval()
    results = []
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels, mask in data_loader:
            inputs = inputs.cuda(0)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(all_labels, all_preds))
        if param == "auc":
            # Multi-class AUC (one-vs-rest)
            try:
                results.append(roc_auc_score(all_labels, all_probs, multi_class='ovr'))
            except:
                results.append(0.0)
        if param == "recall":
            results.append(recall_score(all_labels, all_preds, average='macro'))
        if param == "precision":
            results.append(precision_score(all_labels, all_preds, average='macro'))
        if param == "fmeasure":
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            if precision + recall > 0:
                results.append(2 * precision * recall / (precision + recall))
            else:
                results.append(0.0)
    
    model.train()
    return results, all_preds, all_labels




# =============================================================================
# 2. Core Logic (학습 및 평가 프로세스)
# =============================================================================

def train_single_patch(patch_idx, config):
    """
    하나의 Patch에 대해 학습 -> early stop
    """
    # Config Unpacking
    root_dir = config['root_dir']
    train_files = config['train_files']
    test_files = config['test_files']
    save_dir = config['save_dir']
    time_bin = config['time_bin']
    sampling_rate = config['sampling_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    num_patches = config['num_patches']
    net = config['net']
    criterion = config['criterion']
    optimizer = config['optimzer']

    # 시간 계산
    patch_time_ms = get_patch_time_ms(patch_idx, time_bin, sampling_rate, start_time_ms=-200)
    
    print(f"\n{'='*60}")
    print(f"Processing Train Patch {patch_idx} (Time: {patch_time_ms:.0f}ms)")
    print(f"{'='*60}")

    # 디렉토리 설정
    train_dir = os.path.join(save_dir, f"patch_{patch_idx}")
    os.makedirs(train_dir, exist_ok=True)

    # -----------------------------------------------------
    # A. 데이터 로더 설정 (Train & Validation)
    # -----------------------------------------------------
    # Train Loader
    train_dataset = COMBDataset(
        os.path.join(root_dir, "processed_train"), train_files,
        is_train=True, random_mask=False, time_bin=time_bin, patch_idx=patch_idx
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=False
    )

    # Validation Loader (User Request: test_all 대신 현재 Patch 사용)
    val_dataset = COMBDataset(
        os.path.join(root_dir, "processed_test"), test_files,
        is_train=False, random_mask=False, time_bin=time_bin, patch_idx=patch_idx
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    # -----------------------------------------------------
    # B. 모델 초기화
    # -----------------------------------------------------


    best_acc = 0.0
    best_model_path = None
    early_stop_counter = 0

    # -----------------------------------------------------
    # C. 학습 루프 (Training Loop)
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, mask in train_loader:
            inputs, labels = inputs.cuda(0), labels.cuda(0)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        test_metrics, all_preds, all_labels = evaluate(net, val_loader, ["acc"])
        
        # 0번 제외 Acc 계산 (Early Stopping 기준)
        valid_acc, valid_bal_acc, _, _ = calculate_metrics(all_preds, all_labels)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            early_stop_counter = 0
            
            file_name = f"EEGNet_F1_P{patch_idx}_epoch={epoch+1}_acc={valid_acc:.2f}.pth"
            best_model_path = os.path.join(train_dir, file_name)
            torch.save(net.state_dict(), best_model_path)
            print(f"  [Epoch {epoch+1}] Best Updated: {valid_acc:.2f}% (Loss: {avg_loss:.4f})")
        else:
            early_stop_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
            
            if early_stop_counter >= patience:
                print(f"  [Early Stop] Epoch {epoch+1}")
                break

def test():
    # # -----------------------------------------------------
    # # D. 전체 테스트 루프 (Temporal Generalization)
    # # -----------------------------------------------------
    # # Best Model 로드
    # if best_model_path:
    #     net.load_state_dict(torch.load(best_model_path))
    #     print(f"Loaded Best Model: {os.path.basename(best_model_path)}")

    # results = {
    #     'train_patch_idx': [], 'time_ms': [], 'train_loss': [], 'train_acc': [],
    #     'test_patch_idx': [], 'test_acc': [], 'test_balanced_acc': [],
    #     'test_auc': [], 'test_fmeasure': []
    # }

    # # 모든 Test Patch에 대해 평가 (0 ~ 13)
    # for test_patch_idx in range(num_patches):
    #     # Test Loader 생성
    #     test_dataset_patch = COMBDataset(
    #         os.path.join(root_dir, "processed_test"), test_files,
    #         is_train=False, time_bin=time_bin, patch_idx=test_patch_idx
    #     )
    #     test_loader_patch = torch.utils.data.DataLoader(
    #         test_dataset_patch, batch_size=batch_size, num_workers=0, shuffle=False
    #     )
        
    #     # 평가
    #     test_metrics, all_preds, all_labels = evaluate(net, test_loader_patch, ["acc", "auc", "fmeasure"])
        
    #     # Metrics 계산 (0번 제외)
    #     final_acc, final_bal_acc, filtered_preds, filtered_labels = calculate_metrics(all_preds, all_labels)
        
    #     # Confusion Matrix 저장
    #     cm_filename = f"TrainP{patch_idx}_TestP{test_patch_idx}_Acc{final_acc:.2f}_Bal{final_bal_acc:.2f}_cm.png"
    #     cm_save_path = os.path.join(cm_dir, cm_filename)
    #     cm_title = f"Train P{patch_idx} / Test P{test_patch_idx}\nAcc: {final_acc:.2f}% | Bal: {final_bal_acc:.2f}%"
        
    #     plot_confusion_matrix(filtered_labels, filtered_preds, ['1','2','3','4','5'], cm_save_path, cm_title)
    #     plt.close('all')

    #     # 결과 수집
    #     results['train_patch_idx'].append(patch_idx)
    #     results['time_ms'].append(patch_time_ms)
    #     results['train_loss'].append(avg_loss)
    #     results['train_acc'].append(train_acc)
    #     results['test_patch_idx'].append(test_patch_idx)
    #     results['test_acc'].append(final_acc)
    #     results['test_balanced_acc'].append(final_bal_acc)
    #     results['test_auc'].append(test_metrics[1])
    #     results['test_fmeasure'].append(test_metrics[2])

    # # -----------------------------------------------------
    # # E. 결과 저장 및 정리
    # # -----------------------------------------------------
    # # CSV 저장
    # df_results = pd.DataFrame(results)
    # csv_path = os.path.join(train_dir, f"TrainP{patch_idx}_AllResults.csv")
    # df_results.to_csv(csv_path, index=False)
    # print(f"Results saved to: {csv_path}")

    # # 그래프 저장
    # save_graphs(df_results, train_dir, patch_idx, patch_time_ms)

    # # 메모리 정리
    # del net, optimizer, criterion, train_loader, val_loader
    # if 'test_loader_patch' in locals(): del test_loader_patch
    # torch.cuda.empty_cache()
    # gc.collect()
    # print(f"Finished processing Patch {patch_idx}.")


# =============================================================================
# 3. Main Execution
# =============================================================================

if __name__ == "__main__":
    # 설정 값 (User Configuration)
    net = EEGNet(n_channels=53, n_timepoints=448, n_classes=6).cuda(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    config = {
        'root_dir': "/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG",
        'train_files': os.listdir("/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG/processed_train"),
        'test_files': os.listdir("/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG/processed_test"),
        'save_dir': "/local_raid3/03_user/myyu/EEGNet/logs",
        'time_bin': 32,
        'sampling_rate': 500,
        'num_epochs': 100,
        'batch_size': 64,  # 배치 사이즈가 정의되지 않아 임의로 64 설정 (필요시 수정)
        'patience': 10,
        'num_patches': 448 // 32 # 14,
        'net' : net,
        'criterion' : criterion,
        'optimizer' : optimizer
    }

    # Patch Loop (5부터 시작)
    for patch_idx in range(6, config['num_patches']):
        train_single_patch(patch_idx, config)

    print(f"\n{'='*60}")
    print("All Patches Completed Successfully!")
    print(f"{'='*60}")





time_bin = 32
batch_size = 64
MODEL_INPUT_LEN = 448



# 모델 초기화 (53채널, 448 timepoints, 5 classes)
net = EEGNet(n_channels=53, n_timepoints=448, n_classes=6).cuda(0)
print(f"FC input size: {net.fc_input_size}")

# Loss: CrossEntropyLoss (5-class classification)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())



root_dir = "/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG"
train_files = os.listdir(os.path.join(root_dir, "processed_train"))
test_files = os.listdir(os.path.join(root_dir, "processed_test"))

# print("Loading train dataset...")
# # Training dataset (augmentation 사용)
# train_dataset = COMBDataset(
#     os.path.join(root_dir, "processed_train"), train_files, 
#     strict_mask=True, random_mask=False, time_bin=time_bin
# )
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
# )

# print("\nLoading test dataset...")
# # Test dataset (전체 데이터 사용, patch_idx=None)
# test_dataset = COMBDataset(
#     os.path.join(root_dir, "processed_test"), test_files, 
#     strict_mask=False, time_bin=time_bin
# )
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size, num_workers=8, shuffle=False
# )

# print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# # 데이터 shape 및 레이블 확인
# sample_input, sample_label, sample_mask = train_dataset[2]
# print(f"Sample input shape: {sample_input.shape}")  # (1, 448, 53) expected
# print(f"Sample label (model input, 0-4): {sample_label}")
# print(f"Sample label (display, 1-5): {sample_label + 1}")
# print(f"Sample mask shape: {sample_mask.shape}")

cm_dir = os.path.join(train_dir, "cm")
os.makedirs(cm_dir, exist_ok=True)