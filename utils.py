"""
MemmapLoader 방식 알아보기 -> 원하는 구간만 읽어와서 메모리를 확 줄일 수 있음 -> 우리는 masking이 주된 포인트니까
"""

class COMBDataset(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate, model_input_len, strict_mask=True, random_mask=False, patch_idx=None, time_bin=None):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate
        self.strict_mask = strict_mask
        self.random_mask = random_mask
        self.patch_idx = patch_idx
        self.model_input_len = model_input_len
        if time_bin is not None:
            self.time_bin = time_bin
        else:
            raise ValueError("time_bin must be specified")

        self.total_blocks = self.model_input_len // self.time_bin

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]  # shape: (n_channels, data_len)
        Y = int(sample["label"] - 1)  # [2, 3, 4, 5, 6] -> [0, 1, 2, 3, 4]
        
        n_channels = X.shape[0]
        data_len = X.shape[-1]
        model_input_len = self.model_input_len

        # z-score normalization
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True) + 1e-6
        X = (X - mean) / std

        time_bin = self.time_bin
        max_valid_patch = data_len // time_bin

        # 1. random window augmentation
        if self.random_mask and data_len > time_bin:
            if random.random() < 0.5:
                # cumulative Window Augmentation
                end_patch_idx = random.randint(1, max_valid_patch)
                start_patch_idx = 0
            else:
                # random window augmentation
                end_patch_idx = random.randint(1, max_valid_patch)
                start_patch_idx = random.randint(0, end_patch_idx)
        # 2. Fixed window
        else:
            if self.patch_idx is not None:
                # 지정된 패치 인덱스 사용
                start_patch_idx = self.patch_idx
                end_patch_idx = self.patch_idx + 1  # 끝 인덱스는 포함되지 않으므로 +1
            else:
                # [Test] 전체 사용
                start_patch_idx = 0
                end_patch_idx = max_valid_patch

        # 전체를 0으로 초기화 (보고자 하는 영역만 값이 들어감)
        input_tensor = torch.zeros((n_channels, model_input_len), dtype=torch.float32)
        mask = torch.zeros(model_input_len, dtype=torch.float32)

        # 유효 구간 계산
        start_t_index = start_patch_idx * time_bin
        end_t_index = min(end_patch_idx * time_bin, data_len, model_input_len)

        mask[start_t_index:end_t_index] = 1.0
        input_tensor = X[mask]

        # Shape 변환: (C, T) -> (1, T, C) for EEGNet input
        # EEGNet expects: (batch, 1, T, C)
        input_tensor = input_tensor.transpose(0, 1)  # (T, C)
        input_tensor = input_tensor.unsqueeze(0)     # (1, T, C)

        return input_tensor, Y, mask




def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    """Confusion matrix를 class별 accuracy (확률)로 표시하고 저장"""
    cm_dir = os.path.join(train_dir, "cm")
    os.makedirs(cm_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Row-wise normalization: 각 class별로 맞춘 확률 (recall per class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # NaN 처리
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap with normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    
    # Add raw counts as secondary annotation
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.75, f'({cm[i, j]})', 
                   ha='center', va='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm_normalized


def get_patch_time_ms(patch_idx, time_bin, sampling_rate, start_time_ms=-200):
    """패치 인덱스를 실제 시간(ms)으로 변환"""
    time_per_bin_ms = time_bin / sampling_rate * 1000
    start_ms = patch_idx * time_per_bin_ms + start_time_ms
    return start_ms







def calculate_metrics(all_preds, all_labels):
    """
    0번 클래스를 제외한 1~5번 클래스에 대한 Accuracy와 Balanced Accuracy를 계산합니다.
    """
    np_preds = np.array(all_preds)
    np_labels = np.array(all_labels)
    
    # 0번 클래스(배경/노이즈) 제외
    valid_indices = np.where(np_labels != 0)[0]
    
    if len(valid_indices) > 0:
        filtered_preds = np_preds[valid_indices]
        filtered_labels = np_labels[valid_indices]
        
        acc = 100 * (filtered_preds == filtered_labels).sum() / len(valid_indices)
        bal_acc = 100 * balanced_accuracy_score(filtered_labels, filtered_preds)
    else:
        filtered_preds, filtered_labels = [], []
        acc, bal_acc = 0.0, 0.0
        
    return acc, bal_acc, filtered_preds, filtered_labels

def save_graphs(df_results, save_dir, patch_idx, patch_time_ms):
    """
    Temporal Generalization (Train Patch vs All Test Patches) 그래프를 그립니다.
    """
    plt.figure(figsize=(12, 6))
    
    # Standard Accuracy
    plt.plot(df_results['test_patch_idx'], df_results['test_acc'], 
             marker='o', label='Test Accuracy', color='blue', linewidth=2)
    
    # Balanced Accuracy
    plt.plot(df_results['test_patch_idx'], df_results['test_balanced_acc'], 
             marker='s', label='Balanced Accuracy', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Temporal Generalization: Train on P{patch_idx} ({patch_time_ms:.0f}ms)", fontsize=15)
    plt.xlabel("Test Patch Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(df_results['test_patch_idx'], rotation=45)
    plt.tight_layout()
    
    graph_filename = f"TrainP{patch_idx}_Temporal_Gen_Acc.png"
    graph_path = os.path.join(save_dir, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to: {graph_path}")