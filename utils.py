import os
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score   
from abc import ABC, abstractmethod
import seaborn as sns
import pandas as pd

class BaseFileHandler(ABC):
    """
    핸들러는 데이터 1개씩만 읽어옴.
    파일에서 메타데이터(Shape) 추출과 지정된 구간의 데이터 로드를 담당 
    """
    @abstractmethod
    def get_meta(self, file_path):
        """
        파일에서 메타데이터(Shape)를 추출하는 메서드
        mmap으로 일부만 로딩해서 마스킹하려면 get_meta에서 shape 정보를 알아야 함
        return: tuple: (n, n_channels, data_len) or (n, n_channels, data_len)
        """
        pass

    @abstractmethod
    def load_slice(self, file_path, start_idx, end_idx):
        """
        파일에서 지정된 구간의 데이터를 로드하는 메서드
        return: np.ndarray: shape (n_channels, n_channels, slice_len)
        """
        pass

class NpyFileHandler(BaseFileHandler):
    """
    X_subj: (sum_trials, n_channels, n_times)
    y_subj: (sum_trials,)
    """
    def get_meta(self, file_path):
        data = np.load(file_path, mmap_mode='r')
        return data.shape


    # def _path_info(self, file_dir):
    #     """
    #     부모 경로에서 X, y, stats 파일 이름을 추출하는 메서드
    #     - X_filename = sub-01.npy / sub-01_run01.npy
    #     - y_filename = sub-01_label.npy / sub-01_run01_label.npy
    #     - stats_filename = sub-01_stats.npy / sub-01_run01_stats.npy

    #     return: tuple: (X_filename, y_filename, stats_filename)
    #     """
    #     file_path = os.listdir(file_dir)
    #     X_filename = [f for f in file_path if "label" not in f and "stats" not in f and "info" not in f]
    #     y_filename = [f for f in file_path if "label" in f]
    #     stats_filename = [f for f in file_path if "stats" in f]
    #     infos_filename = [f for f in file_path if "info" in f]

    #     return X_filename, y_filename, stats_filename, infos_filename


    def load_slice(self, file_path, start_idx, end_idx):
    
        """
        file_path = os.listdir(file_path)
        X_filename = [f for f in file_path if "label" not in f]
        y_filename = [f for f in file_path if "label" in f]
        stats_filename = [f for f in file_path if "stats" in f]
        return X_filename, y_filename, stats_filename

        여기서는 window strategy를 받아서 파일 1개씩만 읽어오는 것만 함 

        """
        X_filename = file_path
        y_filename = file_path.replace('.npy', '_label.npy')
        stats_filename = file_path.replace('.npy', '_stats.npy')
        infos_filename = file_path.replace('.npy', '_info.pkl')

        X = np.load(X_filename, mmap_mode ='r')
        y = np.load(y_filename, mmap_mode ='r')
        stat = np.load(stats_filename, mmap_mode ='r')           # (N_trials, n_channels, [mean, std])
        info = pickle.load(open(infos_filename, 'rb'))      # MNE info object

        return X[:, :, start_idx:end_idx], y, stat, info

# !)  수정!!!
class PklFileHandler(BaseFileHandler):
    """
    X_filename = sub-01.pkl / sub-01_run01.pkl
    y_filename = sub-01_label.pkl / sub-01_run01_label.pkl
    X_subj: 
    y_subj: 
    """
    def get_meta(self, file_path):
        pass
    def load_slice(self, file_path, start_idx, end_idx):
        pass

class NpzFileHandler(BaseFileHandler):
    def get_meta(self):
        pass
    def load_slice(self):
        pass


class WindowMaskingStrategy(ABC):
    def __init__(self):
        pass
    def __call__(self, total_len):
        pass


class FixedWindowStrategy(WindowMaskingStrategy):
    def __init__(self, window_len, patch_idx, stride):
        super().__init__()
        self.window_len = window_len
        self.patch_idx = patch_idx
        self.stride = stride

    def __call__(self, total_len):
        step_size = self.stride if self.stride is not None else self.window_len
        start = self.patch_idx * step_size
        end = start + self.window_len
        if end > total_len:
            end = total_len
            start = max(0, end-self.window_len)
            if start >= total_len:
                raise ValueError("Window exceeds total length")
        return start, end


class RandomWindowStrategy(WindowMaskingStrategy):
    def __init__(self):
        super().__init__()

    def __call__(self, total_len):
        start = random.randint(0, total_len - self.window_len)
        end = start + self.window_len
        return start, end



class WindowNormalizing():
    def __init__(self):
        pass

    def __call__(self, X, stat, mask):
        ## (N_trials, n_channels, [mean, std])
        # z-score normalization
        mean = stat[:, :, 0:1] 
        std = stat[:, :, 1:2] 
        X = (X - mean) / std
        X = X * mask
        return X

class WindowMinMaxScaling():
    def __init__(self):
        pass

    def __call__(self, X, stat, mask):
        ## (N_trials, n_channels, [min, max])
        min_val = stat[:, :, 0] 
        max_val = stat[:, :, 1] 
        X = (X - min_val) / (max_val - min_val + 1e-8)
        X = X * mask
        return X

def torch_collate_fn(batch):
    """
    trials 개수가 다르면 오류가 생길 수 있음 collab하는 방식 생각하기
    data_chunk_type이 run일때 subject일때
    """
    pass

class COMBDataset(torch.utils.data.Dataset):
    def __init__(self, config, filepath=None):
        self.data_dir = config["data_dir"]      
        self.sampling_rate = config["sampling_rate"]
        self.window_type = config["window_type"]
        self.patch_idx = config["patch_idx"]
        self.stride = config["stride"]
        self.file_chunk_type = config["file_chunk_type"]

        if filepath is not None:
            self.file_paths = filepath
        else:
            self.file_paths = self._find_data_files(self.data_dir, config["data_ext"])

        if config["time_bin"] is not None:
            self.time_bin = config["time_bin"]
        else:
            raise ValueError("time_bin must be specified")

        if config["data_ext"] =='npy':
            self.file_handler = NpyFileHandler()
        elif config["data_ext"] == 'pkl':
            self.file_handler = PklFileHandler()
        elif config["data_ext"] == 'npz':
            self.file_handler = NpzFileHandler()
        else:
            raise ValueError(f"Unsupported data extension: {config['data_ext']}")

        if self.file_chunk_type == "subject":
            pass
        elif self.file_chunk_type == "run":
            pass
        else:
            raise ValueError(f"Unsupported file chunk type: {self.file_chunk_type}")


        if config["window_type"] == "fixed":
            self.window_type = FixedWindowStrategy(window_len = self.time_bin, patch_idx = self.patch_idx, stride = self.stride)
        elif config["window_type"] == "random":
            self.window_type = RandomWindowStrategy()
        else:
            raise ValueError(f"Unsupported window type: {config['window_type']}")

        if config["normalize_method"] == "zscore":
            self.normalizer = WindowNormalizing()
        elif config["normalize_method"] == "minmax":
            self.normalizer = WindowMinMaxScaling()
        else:
            raise ValueError(f"Unsupported normalization method: {config['normalize_method']}")

        self.model_input_len, self.model_input_ch = self._get_sample_info()

    def _find_data_files(self, data_dir, data_ext):
        """
        지정된 디렉토리에서 특정 확장자를 가진 데이터 파일들을 찾는 함수
        return: list of file paths
        """
        all_files = glob.glob(os.path.join(data_dir, f"*.{data_ext}"))
        main_files = [
            f for f in all_files 
            if "label" not in f and "stats" not in f and "info" not in f
        ]
        main_files.sort()
        return main_files

    def _get_sample_info(self):
        """
        파일에서 샘플 메타데이터(Shape)를 추출하는 메서드
        return: tuple: (n, n_channels, data_len)
        """

        if not self.file_paths:
            raise ValueError("File list is empty!")
        
        sample_path = self.file_paths[0]
        
        # shape: (N_trials, N_channels, N_times)
        try:
            shape = self.file_handler.get_meta(sample_path)
            total_time = shape[-1] 
            # print(f"[Info] Auto-detected total time length: {total_time}")
            ch = shape[1]
            return total_time, ch
        except Exception as e:
            raise RuntimeError(f"Failed to read sample file {sample_path}: {e}")

    def _masking_from_window(self, total_len, X, start_idx, end_idx):

        n_trials = X.shape[0]
        n_channels = X.shape[1]

        mask = np.zeros(total_len, dtype=bool)
        mask[start_idx:end_idx] = True
        mask_expand = np.tile(mask[np.newaxis, np.newaxis, :], (n_trials, n_channels, 1))

        full_X = np.zeros((X.shape[0], X.shape[1], total_len))
        full_X[:, :, start_idx:end_idx] = X

        return mask_expand, full_X


    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, index):
        """
        getitem 메서드는 데이터셋에서 특정 인덱스의 데이터를 가져오는 역할을 합니다.
        dataset의 필수 함수. 이걸 dataloader가 호출함.
        """
        path = self.file_paths[index] 
        start, end = self.window_type(total_len=self.model_input_len)
        X, Y, stat, _ = self.file_handler.load_slice(path, start, end)
        mask, X = self._masking_from_window(self.model_input_len, X, start, end)
        X = self.normalizer(X, stat, mask)

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long() 
        stat = torch.from_numpy(stat).float()
        mask = torch.from_numpy(mask).bool() # BoolTensor로 변환

        # print(f"\n[File Inspection] Loading: {os.path.basename(path)}")
        # print(f"   - X shape (Loaded): {X.shape}")
        # print(f"   - Y shape (Loaded): {Y.shape}")

        return X, Y, stat, mask

class InferenceManager:
    def __init__(self, config, model):
        pass
    def __call__(self, *args, **kwds):
        pass
    def _load_model(self, patch_idx):
        if config["model_load_from"]:
            self.model.ckpt_load(config["model_load_from"])
        pass

    def _create_loader(self, patch_idx):
        pass
    def run_all_patches(self):
        pass

class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def ckpt_load(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # 체크포인트에 'model_state_dict'가 있으면 해당 상태를 로드
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 체크포인트가 'model_state_dict'를 포함하지 않으면 전체 체크포인트를 로드
            self.model.load_state_dict(checkpoint)

    def predict(self, model, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():   
            for batch_x, batch_y, _, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        return all_preds, all_labels
        

class Evaluator:
    def __init__(self):
        pass
    def compute_metrics(self, all_preds, all_labels):
        pass

class Recorder:
    def __init__(self):
        pass
    def save_detailed_results(self, info_dict, csv_save_path, preds, labels, prob = None):
        csv_dir = os.path.dirname(csv_save_path)
        os.makedirs(csv_dir, exist_ok=True)
        df = pd.DataFrame(info_dict)
        df.to_csv(csv_save_path, index=False)

        # Save additional prediction details
        preds_df = pd.DataFrame({'preds': preds, 'labels': labels})
        preds_df.to_csv(os.path.join(csv_dir, "predictions.csv"), index=False)

        if prob is not None:
            prob_df = pd.DataFrame(prob)
            prob_df.to_csv(os.path.join(csv_dir, "probabilities.csv"), index=False)

class Visualizer:
    def __init__(self, base_dir):
        pass
    def plot_cm(self, labels, preds, cm_save_path, cm_title):
        cm_dir = os.path.dirname(cm_save_path)
        os.makedirs(cm_dir, exist_ok=True)
        pass
    def plot_tgm(self):
        pass
    
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

class ResultLoader:
    def load(self):
        pass


def get_patch_time_ms(patch_idx, time_bin, sampling_rate, start_time_ms=-200):
    """패치 인덱스를 실제 시간(ms)으로 변환"""
    time_per_bin_ms = time_bin / sampling_rate * 1000
    start_ms = patch_idx * time_per_bin_ms + start_time_ms
    end_ms = start_ms + time_per_bin_ms
    return start_ms, end_ms

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
        # bal_acc = 100 * balanced_accuracy_score(filtered_labels, filtered_preds )
        bal_acc = 100 * recall_score(
                                    filtered_labels, 
                                    filtered_preds, 
                                    average='macro',       # Macro Average Recall = Balanced Accuracy
                                    zero_division=0.0, 
                                    labels=np.unique(filtered_labels)   # 참고: "다중 클래스 분류"에서 사용
                                    )
    else:
        filtered_preds, filtered_labels = [], []
        acc, bal_acc = 0.0, 0.0
        
    return acc, bal_acc, np_preds, np_labels

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


def plot_confusion_matrix(labels, preds, cm_save_path, cm_title):
    label_names = np.unique(labels)

    cm = confusion_matrix(labels, preds, labels=label_names)

    # Train/Test 시간 계산 (matched patch)
    # train_time_start = START_TIME + train_patch_idx * TIME_BIN * 1000 / SAMPLING_RATE
    # train_time_end = START_TIME + (train_patch_idx + 1) * TIME_BIN * 1000 / SAMPLING_RATE

    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=[f"Class_{i}" for i in label_names],
                yticklabels=[f"Class_{i}" for i in label_names])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title(cm_title)
    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=150)
    plt.close('all')

def torch_collate_fn(batch):
    """
    trials 개수가 다르면 오류가 생길 수 있음 collab하는 방식 생각하기
    data_chunk_type이 run일때 subject일때
    """
    batch_x, batch_y, batch_stat, batch_mask = zip(*batch)
    
    X = torch.cat(batch_x, axis=0)
    y = torch.cat(batch_y, axis=0)
    stat = torch.cat(batch_stat, axis=0)
    mask = torch.cat(batch_mask, axis=0)

    return X, y, stat, mask



def build_tgm_matrix(all_model_metrics, metric_key="accuracy"):
    """
    [기능]
    - 여러 모델(각기 다른 시간대에서 학습)의 메트릭을 모아 TGM Matrix 생성

    [입력]
    - all_model_metrics: dict 형태 {train_patch_idx: metrics_dict, ...}
        - metrics_dict는 compute_metrics()의 출력
    - metric_key: TGM에 사용할 메트릭 ("accuracy", "balanced_accuracy", "f1_macro", etc.)

    [출력]
    - tgm_matrix: np.ndarray [num_train_patches, num_test_patches]
    - train_indices: 학습 패치 인덱스 리스트
    - test_indices: 테스트 패치 인덱스 리스트
    """
    # 학습 패치 인덱스 정렬
    train_indices = sorted(all_model_metrics.keys())

    # 테스트 패치 인덱스는 첫 번째 모델의 per_patch에서 가져옴
    first_metrics = all_model_metrics[train_indices[0]]
    test_indices = [p["patch_idx"] for p in first_metrics["per_patch"]]

    # TGM Matrix 초기화
    num_train = len(train_indices)
    num_test = len(test_indices)
    tgm_matrix = np.zeros((num_train, num_test))

    # Matrix 채우기
    for i, train_idx in enumerate(train_indices):
        metrics = all_model_metrics[train_idx]
        for j, patch_result in enumerate(metrics["per_patch"]):
            tgm_matrix[i, j] = patch_result[metric_key]

    return tgm_matrix, train_indices, test_indices


def plot_tgm_heatmap(config, tgm_matrix,
                     metric_name="Accuracy", save_path="tgm_heatmap.png",
                     vmin=None, vmax=None, cmap="viridis"):
    """
    [기능]
    - TGM Matrix를 Heatmap으로 시각화

    [입력]
    - tgm_matrix: np.ndarray [num_train_patches, num_test_patches]
    - train_indices: 학습 패치 인덱스 리스트
    - test_indices: 테스트 패치 인덱스 리스트
    - sampling_rate: 샘플링 레이트 (Hz)
    - time_bin: 시간 빈 크기 (samples)
    - start_time: 시작 시간 (ms)
    - metric_name: 메트릭 이름 (제목용)
    - save_path: 저장 경로
    - vmin, vmax: 컬러맵 범위
    - cmap: 컬러맵 종류

    [출력]
    - fig: matplotlib figure 객체
    """
    # # 시간 레이블 생성 (ms 단위)
    # def patch_to_time_ms(patch_idx):
    #     return start_time + patch_idx * time_bin * 1000 / sampling_rate

    # train_labels = [f"P{idx}_{patch_to_time_ms(idx):.0f}" for idx in train_indices]
    # test_labels = [f"P{idx}_{patch_to_time_ms(idx):.0f}" for idx in test_indices]

    train_start_time, train_end_time = get_patch_time_ms(patch_idx, config["time_bin"], config["sampling_rate"], start_time_ms=config["start_time_ms"])
    test_start_time, test_end_time = get_patch_time_ms(patch_idx, config["time_bin"], config["sampling_rate"], start_time_ms=config["start_time_ms"])

    # Figure 생성
    fig, ax = plt.subplots(figsize=(12, 10))

    # Heatmap 그리기
    sns.heatmap(
        tgm_matrix,
        annot=True if tgm_matrix.shape[0] <= 10 else False,  # 작은 matrix만 숫자 표시
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=test_labels,
        yticklabels=train_labels,
        ax=ax,
        cbar_kws={"label": metric_name}
    )

    # 대각선 표시 (Train Time == Test Time)
    ax.plot([0, min(len(test_indices), len(train_indices))],
            [0, min(len(test_indices), len(train_indices))],
            'r--', linewidth=2, alpha=0.7)

    ax.set_xlabel(f"Test Time (ms) [{test_time_start:.0f} ~ {test_time_end:.0f}ms]", fontsize=12)
    ax.set_ylabel(f"Train Time (ms) [{train_time_start:.0f} ~ {train_time_end:.0f}ms]", fontsize=12)
    ax.set_title(f"Temporal Generalization Matrix ({metric_name})\n"
                 f"Train: {train_time_start:.0f}~{train_time_end:.0f}ms | Test: {test_time_start:.0f}~{test_time_end:.0f}ms",
                 fontsize=14)

    # X축 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[*] TGM Heatmap saved at: {save_path}")

    return fig

# # data inference check!
if __name__ == "__main__":
    config = {
                "data_dir": "./EEG(500Hz)_COMB/processed_test/npy",
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
                "is_label_null": True,
                "skip_inference" : False,
                "metrics":["acc", "bal_acc"]

    } 

    train_files = ['./EEG(500Hz)_COMB/processed_t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          rain/npy/sub-14.npy']
    
    input_len, input_ch = COMBDataset(config=config, filepath = train_files)._get_sample_info()
    # # 모델 초기화 (53채널, 448 timepoints, 5 classes)
    net = EEGNet(n_channels=input_ch, n_timepoints=input_len, n_classes=config["n_classes"]).cuda(0)
    print(f"FC input size: {net.fc_input_size}")

    # Loss: CrossEntropyLoss (5-class classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    n_patches = input_len//config["time_bin"]



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

