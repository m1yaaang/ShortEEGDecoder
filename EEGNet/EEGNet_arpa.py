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
import logging
from tqdm import tqdm
from utils_my import COMBDataset, calculate_metrics, get_patch_time_ms, torch_collate_fn, plot_confusion_matrix, save_graphs
from sklearn.model_selection import KFold

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
        if x.ndim == 3:
            x = x.unsqueeze(1).permute(0, 1, 3, 2)      # (B, 1, T, C)

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = x.permute(0, 3, 1, 2)                   # (?, C, B, T)
        
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
        for inputs, labels, _, mask in data_loader:
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
            results.append(precision_score(all_labels, all_preds, average='macro', zero_division=0.0))
        if param == "fmeasure":
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0.0, labels=np.unique(all_labels))
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0.0, labels=np.unique(all_labels))
            if precision + recall > 0:
                results.append(2 * precision * recall / (precision + recall))
            else:
                results.append(0.0)
    
    model.train()
    return results, all_preds, all_labels




# =============================================================================
# 2. Core Logic (학습 및 평가 프로세스)
# =============================================================================

def train_single_patch(config, net, criterion, optimizer, train_loader, val_loader):
    """
    하나의 Patch에 대해 학습 -> early stop
    """
    # # Config Unpacking 
    num_epochs = config['num_epochs']
    patch_idx = config['patch_idx']
    patience = config['patience']


    # -----------------------------------------------------
    # B. 모델 초기화
    # -----------------------------------------------------
    best_acc = 0.0
    best_model_path = None
    early_stop_counter = 0

    # 디렉토리 설정
    train_dir = os.path.join(config["save_dir"], f"patch_{config['patch_idx']}")
    os.makedirs(train_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # -----------------------------------------------------
    # C. 학습 루프 (Training Loop)
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, (inputs, labels, _, _) in enumerate(loop):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        test_metrics, all_preds, all_labels = evaluate(net, val_loader, ["acc"])
        
        # 0번 제외 Acc 계산 (Early Stopping 기준)
        valid_acc, valid_bal_acc, _, _ = calculate_metrics(all_preds, all_labels)

        logger.info(f"  [Epoch {epoch+1}] Valid Acc: {valid_acc:.2f}% | Valid Bal Acc: {valid_bal_acc:.2f}%")

        if valid_acc > best_acc:
            best_acc = valid_acc
            early_stop_counter = 0
            
            file_name = f"EEGNet_F1_P{patch_idx}_epoch={epoch+1}_acc={valid_acc:.2f}.pth"
            best_model_path = os.path.join(train_dir, file_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, best_model_path)

            print(f"  [Epoch {epoch+1}] Best Updated: {valid_acc:.2f}% (Loss: {avg_loss:.4f})")
        else:
            early_stop_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
            
            if early_stop_counter >= patience:
                print(f"  [Early Stop] Epoch {epoch+1}")
                break

def test(config, net, ckpt_paths, num_patches):

    test_acc_list = []
    df_results = pd.DataFrame()

    # -----------------------------------------------------
    # D. 전체 테스트 루프 (Temporal Generalization)
    # -----------------------------------------------------
    # Best Model 로드
    for ckpt in ckpt_paths: 
        checkpoint = torch.load(ckpt)
        # 체크포인트에 'model_state_dict'가 있으면 해당 상태를 로드
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 체크포인트가 'model_state_dict'를 포함하지 않으면 전체 체크포인트를 로드
            net.load_state_dict(checkpoint)
        net.eval()
        print(f"Loaded Best Model: {ckpt}")
        train_patch_idx = int(ckpt.split("patch_")[1].split("/")[0])
        train_epoch = int(ckpt.split("epoch=")[1].split("_")[0])

        base_dir = os.path.dirname(ckpt)
        cm_dir = os.path.join(base_dir, "cm")
        csv_dir = os.path.join(base_dir, "csv")
        os.makedirs(cm_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, config['time_bin'], config['sampling_rate'])

        print(f"Test Patch{train_patch_idx}({train_start_ms}~{train_end_ms})")
        results = {
            'train_patch_idx': [], 'train_time_ms': [], 
            'test_patch_idx': [], 'test_time_ms': [], 'test_acc': [], 'test_balanced_acc': [],
            'test_auc': [], 'test_fmeasure': []
        }

        # 모든 Test Patch에 대해 평가 (0 ~ 13)
        for test_patch_idx in range(num_patches):
            # Test Loader 생성
            test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, config['time_bin'], config['sampling_rate'])
            print(f"Test Patch{test_patch_idx}({test_start_ms}~{test_end_ms})")
            config["patch_idx"] = test_patch_idx
            test_patch_dataset = COMBDataset(config=config)
            test_patch_loader = torch.utils.data.DataLoader(
                                                test_patch_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                num_workers=config["num_workers"],
                                                collate_fn = torch_collate_fn,)
            # print(f"Found Files: {len(test_patch_dataset)} files")

            # 평가
            test_metrics, all_preds, all_labels = evaluate(net, test_patch_loader, ["acc", "auc", "fmeasure"])
            
            # Metrics 계산 (0번 제외)
            final_acc, final_bal_acc, filtered_preds, filtered_labels = calculate_metrics(all_preds, all_labels)
            
            # Confusion Matrix 저장
            cm_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_epoch{train_epoch}_Acc{final_acc:.2f}_Bal{final_bal_acc:.2f}_cm.png"
            cm_save_path = os.path.join(cm_dir, cm_filename)
            cm_title = f"Train P{train_patch_idx} / Test P{test_patch_idx}\nAcc: {final_acc:.2f}% | Bal: {final_bal_acc:.2f}%"

            plot_confusion_matrix(filtered_labels, filtered_preds, cm_save_path, cm_title)

            results_detail = {
                'train_patch_idx': train_patch_idx,
                'train_time_ms': train_start_ms,
                'test_patch_idx': test_patch_idx,
                'test_time_ms': test_start_ms,
                'prediction': all_preds,
                'label': all_labels,
            }
            csv_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_epoch{train_epoch}_results.csv"
            csv_save_path = os.path.join(csv_dir, csv_filename)
            df_detail = pd.DataFrame(results_detail)
            df_detail.to_csv(csv_save_path, index=False)

            # 결과 수집
            results['train_patch_idx'].append(train_patch_idx)
            results['train_time_ms'].append(train_start_ms)
            results['test_patch_idx'].append(test_patch_idx)
            results['test_time_ms'].append(test_start_ms)
            results['test_acc'].append(final_acc)
            results['test_balanced_acc'].append(final_bal_acc)
            results['test_auc'].append(test_metrics[1])
            results['test_fmeasure'].append(test_metrics[2])

            del test_patch_loader, test_patch_dataset
            torch.cuda.empty_cache()
            gc.collect()

        test_acc_list.append(pd.DataFrame(results))

        # -----------------------------------------------------
        # E. 결과 저장 및 정리
        # -----------------------------------------------------
        # CSV 저장
        df_results = pd.concat(test_acc_list, ignore_index=True)
        csv_path = os.path.join(csv_dir, f"TrainP{train_patch_idx}_AllResults.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")



    # # 그래프 저장
    # save_graphs(df_results, train_dir, patch_idx, patch_time_ms)

    # # 메모리 정리
    del net,
    torch.cuda.empty_cache()
    gc.collect()

    # tgm matrix 만들기 
    plot_tgm_from_df(df_results, metric_key='test_acc', save_dir=base_dir)


def plot_tgm_from_df(df, metric_key='test_acc', save_dir='./'):
    """
    [기능]
    - DataFrame의 모든 숫자 데이터(idx, ms)를 이용해 TGM Heatmap을 그립니다.
    - 문자열 컬럼(_str)이 없어도 동작하며, 모든 라벨을 전부 표시합니다.
    """
    
    # 1. Pivot Table (Matrix 변환)
    tgm_matrix = df.pivot(index='train_patch_idx', columns='test_patch_idx', values=metric_key)
    matrix_values = tgm_matrix.values
    
    # 인덱스 리스트 (정렬됨)
    train_indices = tgm_matrix.index.tolist()
    test_indices = tgm_matrix.columns.tolist()

    # 2. 라벨 생성 (데이터프레임 내 ms 정보가 있으면 활용)
    
    # (1) Train Label 생성
    if 'train_time_ms' in df.columns:
        train_map = df[['train_patch_idx', 'train_time_ms']].drop_duplicates().set_index('train_patch_idx')
        train_labels = [f"P{idx} ({train_map.loc[idx, 'train_time_ms']:.0f}ms)" for idx in train_indices]
    else:
        train_labels = [f"P{idx}" for idx in train_indices]

    # (2) Test Label 생성
    if 'test_time_ms' in df.columns:
        test_map = df[['test_patch_idx', 'test_time_ms']].drop_duplicates().set_index('test_patch_idx')
        test_labels = [f"P{idx} ({test_map.loc[idx, 'test_time_ms']:.0f}ms)" for idx in test_indices]
    else:
        test_labels = [f"P{idx}" for idx in test_indices]

    # 3. Plotting
    plt.figure(figsize=(15, 13)) # 라벨이 많으므로 그림 크기를 넉넉하게 잡음
    
    ax = sns.heatmap(
        matrix_values,
        annot=False,            # 칸이 빽빽하면 숫자가 겹치므로 False 권장 (필요시 True)
        fmt=".1f",
        cmap="viridis",
        xticklabels=test_labels, # [수정] 모든 라벨 리스트를 직접 전달
        yticklabels=train_labels, # [수정] 모든 라벨 리스트를 직접 전달
        cbar_kws={"label": metric_key}
    )

    # 4. 축 라벨 스타일 설정 (모든 라벨 표시)
    plt.title(f"Temporal Generalization Matrix ({metric_key})", fontsize=18, pad=20)
    plt.xlabel("Test Time", fontsize=14)
    plt.ylabel("Train Time", fontsize=14)

    # X축 라벨: 45도 회전, 폰트 사이즈 조절
    plt.xticks(rotation=45, ha='right', fontsize=9) 
    
    # Y축 라벨: 0도 (가로) 유지, 폰트 사이즈 조절
    plt.yticks(rotation=0, fontsize=9)

    # 5. 대각선 (Train == Test 시점)
    plt.plot([0, len(test_labels)], [0, len(train_labels)], 
             color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(save_dir, f"TGM_{metric_key}_full.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[*] TGM Heatmap (All Labels) saved at: {save_path}")


# =============================================================================
# 3. Main Execution
# =============================================================================


if __name__ == "__main__":
    config = {
                "data_dir": "./EEG(500Hz)_COMB",
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": True,
                "sampling_rate": 500,
                "start_time_ms" : -200,
                "data_ext": "npy",
                "window_type": "fixed",  # "fixed" or "random"
                "time_bin": 32,
                "file_chunk_type": "subject", # "subject" or "run"
                "normalize_method": "zscore", # "zscore" or "minmax"
                "patch_idx": None,
                "stride": None,
                "save_dir": "EEGNet/logs",
                "num_epochs": 100,
                "patience": 10,
                "n_classes": 6,
    } 

    # print(f"Detected sample shape: Channels={input_ch}, Timepoints={input_len}")

    train_dir = os.path.join(config["data_dir"], "processed_train/npy")

    train_files = [
            os.path.join(train_dir, f) for f in os.listdir(train_dir) 
            if "label" not in f and "stats" not in f and "info" not in f
        ]

    input_len, input_ch = COMBDataset(config=config, filepath = train_files)._get_sample_info()
    # # 모델 초기화 (53채널, 448 timepoints, 5 classes)
    net = EEGNet(n_channels=input_ch, n_timepoints=input_len, n_classes=config["n_classes"]).cuda(0)
    print(f"FC input size: {net.fc_input_size}")

    # Loss: CrossEntropyLoss (5-class classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    n_patches = input_len//config["time_bin"]
    
    k_folds = 2 
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # ------------------config---------------------------



    # 폴드 결과 저장
    fold_results = []

    # -------------------------------------------------------
    # Loop 1: K fold
    # -------------------------------------------------------
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_files)):
    # for fold in range(1):
        
        current_fold = fold + 1
        print(f"\n" + "="*40)
        print(f"[:] Starting Fold {current_fold}/{k_folds}")
        print("="*40)

        # Patch Loop (5부터 시작)
        for patch_idx in range(n_patches):

            train_files_fold = [train_files[i] for i in train_ids]
            val_files_fold = [train_files[i] for i in val_ids]
            
            config["patch_idx"] = patch_idx

            train_dataset = COMBDataset(config=config, file_path = train_files_fold)
            train_loader = torch.utils.data.DataLoader(
                                                train_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=config["shuffle"], 
                                                num_workers=config["num_workers"],
                                                collate_fn = torch_collate_fn,)

            # test_config = config.copy()
            # test_config["data_dir"] = "./EEG(500Hz)_COMB/processed_test/npy"

            val_dataset = COMBDataset(config=config, file_path = val_files_fold)
            val_loader = torch.utils.data.DataLoader(
                                                val_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=config["shuffle"], 
                                                num_workers=config["num_workers"],
                                                collate_fn = torch_collate_fn,)


            train_single_patch(config, net, criterion, optimizer, train_loader, val_loader)

    # -------------------------------------------------------
    # Loop 2: Test all best models
    # -------------------------------------------------------

    ckpt_dir = config["save_dir"]
    best_model_paths = []
    ckpt_lists = os.listdir(ckpt_dir)
    ckpt_lists = [d for d in ckpt_lists if os.path.isdir(os.path.join(ckpt_dir, d))]
    ckpt_lists.sort()

    for p in ckpt_lists:
        if "None" in p:
            continue
        ckpt_best_acc = -1.0
        ckpt_best_file = None
        patch_dir = os.path.join(ckpt_dir, p)
        ckpts = [f for f in os.listdir(patch_dir) if f.endswith(".pth")]
        for c in ckpts:
            try:
                c_acc = float(c.split("_acc=")[-1].replace(".pth",""))
                if c_acc > ckpt_best_acc:
                    ckpt_best_acc = c_acc
                    ckpt_best_file = c
            except:
                continue
        best_model_paths.append(os.path.join(patch_dir, ckpt_best_file))
        
    test_config = config.copy()
    test_config["data_dir"] = "./EEG(500Hz)_COMB/processed_test/npy"
    test(test_config,net, best_model_paths, n_patches)
