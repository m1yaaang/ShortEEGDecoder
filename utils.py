import os
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score   
from abc import ABC, abstractmethod

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


    def _path_info(self, file_path):
        """
        부모 경로에서 X, y, stats 파일 이름을 추출하는 메서드
        - X_filename = sub-01.npy / sub-01_run01.npy
        - y_filename = sub-01_label.npy / sub-01_run01_label.npy
        - stats_filename = sub-01_stats.npy / sub-01_run01_stats.npy

        return: tuple: (X_filename, y_filename, stats_filename)
        """
        file_path = os.listdir(file_path)
        X_filename = [f for f in file_path if "label" not in f and "stats" not in f]
        y_filename = [f for f in file_path if "label" in f]
        stats_filename = [f for f in file_path if "stats" in f]

        return X_filename, y_filename, stats_filename


    def load_window(self, file_path, start_idx, end_idx):
    
        """
        file_path = os.listdir(file_path)
        X_filename = [f for f in file_path if "label" not in f]
        y_filename = [f for f in file_path if "label" in f]
        stats_filename = [f for f in file_path if "stats" in f]
        return X_filename, y_filename, stats_filename

        여기서는 window strategy를 받아서 파일 1개씩만 읽어오는 것만 함 

        """
        X_filename, y_filename, stats_filename = self._path_info(file_path)
        X = np.load(X_filename, mode ='r')
        y = np.load(y_filename, mode ='r')

        n, n_channels, data_len = self.get_meta(X_filename)
        n = X.shape[0]
        n_channels = X.shape[1]
        data_len = X.shape[-1]

        return X[:, start_idx:end_idx]

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


def torch_collate_fn(batch):
    """
    trials 개수가 다르면 오류가 생길 수 있음 collab하는 방식 생각하기
    data_chunk_type이 run일때 subject일때
    """
    pass

class COMBDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, config):
        self.root = config.root
        self.files = config.files
        self.sampling_rate = config.sampling_rate
        self.strict_mask = config.strict_mask
        self.random_mask = config.random_mask
        self.patch_idx = config.patch_idx
        self.model_input_len = config.model_input_len
        self.data_ext = config.data_ext

        if config.time_bin is not None:
            self.time_bin = config.time_bin
        else:
            raise ValueError("time_bin must be specified")

        # self.total_blocks = self.model_input_len // self.time_bin

        if config.ext =='npy':
            self.file_handler = NpyFileHandler()
        elif config.ext == 'pkl':
            self.file_handler = PklFileHandler()
        elif config.ext == 'npz':
            self.file_handler = NpzFileHandler()
        else:
            raise ValueError(f"Unsupported data extension: {config.ext}")


        self.file_chunk_type = config.file_chunk_type
        if self.file_chunk_type == "subject":
            pass
        elif self.file_chunk_type == "run":
            pass
        else:
            raise ValueError(f"Unsupported file chunk type: {self.file_chunk_type}")


        if config.window_type == "fixed":
            self.window_type = FixedWindowStrategy()
        elif config.window_type == "random":
            self.window_type = RandomWindowStrategy()
        else:
            raise ValueError(f"Unsupported window type: {config.window_type}")
        


        self.file_paths = filepath

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        getitem 메서드는 데이터셋에서 특정 인덱스의 데이터를 가져오는 역할을 합니다.
        dataset의 필수 함수. 이걸 dataloader가 호출함.
        """
        path = self.file_paths[index]
        # Implement data retrieval logic here
        raw_signal, labels, stats = self.handler.load_window(path, start, end)
        if self.data_ext == "npy":
            sample = np.load(os.path.join(self.root, self.file_paths[index]), allow_pickle=True)

            X = sample["signal"]  # shape: (n_channels, data_len)
            Y = int(sample["label"])  # [0, 1, 2, 3, 4,5]
        elif self.data_ext == "pkl":
            sample = pickle.load(open(os.path.join(self.root, self.file_paths[index]), "rb"))
        else:
            raise ValueError(f"Unsupported data extension: {self.data_ext}")

        X = sample["signal"]  # shape: (n_channels, data_len)
        Y = int(sample["label"])  # [2, 3, 4, 5, 6] -> [0, 1, 2, 3, 4]

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
                start_patch_idx = random.randint(0, end_patch_idx - 1)

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




# data loader check!
    if __name__=="__main__":
        config = {
            "data_dir": "./data",
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True,
            "ext": "npy",
            "window_type": "fixed",  # "fixed" or "random"
            "input_len": 1000,
            "time_bin": 50,
            "file_chunk_type": "subject" # "subject" or "run"
            
        }