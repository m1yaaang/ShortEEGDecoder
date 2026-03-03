import os
import random
import pickle
import numpy as np
import torch
import glob
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score   
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


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

    def get_info(self, file_path):

        infos_filename = file_path.replace('.npy', '_info.pkl')
        info = pickle.load(open(infos_filename, 'rb'))      # MNE info object
        return info

    def load_slice(self, file_path, trial_idx, start_idx, end_idx):

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

        X = np.load(X_filename, mmap_mode ='r')
        y = np.load(y_filename, mmap_mode ='r')
        stat = np.load(stats_filename, mmap_mode ='r')           # (N_trials, n_channels, [mean, std])

        return X[trial_idx, :, start_idx:end_idx], y[trial_idx], stat[trial_idx]

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

        self.trial_map = []
        self._build_trial_map()

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


    def _build_trial_map(self):
        """Build trial map (file_index, trial_index) for data loading"""
        # print("[*] Building Trial Map...")
        for f_idx, path in enumerate(self.file_paths):
            try:
                shape = self.file_handler.get_meta(path)
                n_trials = shape[0] # (N_trials, Ch, Time)
                
                for t_idx in range(n_trials):
                    self.trial_map.append((f_idx, t_idx))
            except Exception as e:
                print(f"[Warning] Error reading {path}: {e}")
        
        # print(f"[*] Total Trials Indexed: {len(self.trial_map)}")

    def __len__(self):
        return len(self.trial_map)


    def __getitem__(self, index):
        """
        getitem 메서드는 데이터셋에서 특정 인덱스의 데이터를 가져오는 역할을 합니다.
        dataset의 필수 함수. 이걸 dataloader가 호출함.
        """
        file_idx, trial_idx = self.trial_map[index]
        path = self.file_paths[file_idx]

        start, end = self.window_type(total_len=self.model_input_len)
        X, Y, stat = self.file_handler.load_slice(path, trial_idx, start, end)

        X = X[np.newaxis, :, :]        # (1, Ch, Time)
        stat = stat[np.newaxis, :, :]  # (1, Ch, 2)

        mask, X = self._masking_from_window(self.model_input_len, X, start, end)
        X = self.normalizer(X, stat, mask)

        X = torch.from_numpy(X).float()
        Y = torch.tensor(Y).long().unsqueeze(0)
        stat = torch.from_numpy(stat.copy()).float()
        mask = torch.from_numpy(mask).bool() # BoolTensor로 변환

        # print(f"\n[File Inspection] Loading: {os.path.basename(path)}")
        # print(f"   - X shape (Loaded): {X.shape}")
        # print(f"   - Y shape (Loaded): {Y.shape}")

        return X, Y, stat, mask


class WithinSubjectDatasetTotal(COMBDataset):
    """
    COMBDataset을 상속받아 trial 단위로 train/test split을 지원하는 Dataset.
    Multi-file subject에서도 global trial index를 올바르게 처리.
    데이터를 메모리에 캐싱하여 I/O 병목 제거.
    """
    def __init__(self, config, filepath=None, trial_indices=None, shared_cache=None):
        self.selected_trial_indices = trial_indices
        self._cache = shared_cache if shared_cache is not None else {}
        super().__init__(config, filepath)
        self._preload_cache()

    def _preload_cache(self):
        """모든 파일의 mmap 참조를 캐싱 (매 __getitem__마다 np.load 호출 방지)"""
        for path in self.file_paths:
            if path not in self._cache:
                y_path = path.replace('.npy', '_label.npy')
                stat_path = path.replace('.npy', '_stats.npy')
                self._cache[path] = (
                    np.load(path, mmap_mode='r'),
                    np.load(y_path, mmap_mode='r'),
                    np.load(stat_path, mmap_mode='r'),
                )

    def __getitem__(self, index):
        """캐싱된 mmap에서 직접 읽어 I/O 오버헤드 제거"""
        file_idx, trial_idx = self.trial_map[index]
        path = self.file_paths[file_idx]

        start, end = self.window_type(total_len=self.model_input_len)

        X_all, Y_all, stat_all = self._cache[path]
        X = X_all[trial_idx, :, start:end]
        Y = Y_all[trial_idx]
        stat = stat_all[trial_idx]

        X = X[np.newaxis, :, :]
        stat = stat[np.newaxis, :, :]

        mask, X = self._masking_from_window(self.model_input_len, X, start, end)
        X = self.normalizer(X, stat, mask)

        X = torch.from_numpy(X).float()
        Y = torch.tensor(Y).long().unsqueeze(0)
        stat = torch.from_numpy(stat.copy()).float()
        mask = torch.from_numpy(mask).bool()

        return X, Y, stat, mask

    def _build_trial_map(self):
        """
        Global index 기반 trial_map 빌드.
        split_trials가 생성한 global indices를 (file_idx, local_trial_idx)로 변환.
        """
        global_offset = 0
        for f_idx, path in enumerate(self.file_paths):
            try:
                shape = self.file_handler.get_meta(path)
                n_trials = shape[0]

                if self.selected_trial_indices is not None:
                    for global_t_idx in self.selected_trial_indices:
                        local_idx = global_t_idx - global_offset
                        if 0 <= local_idx < n_trials:
                            self.trial_map.append((f_idx, local_idx))
                else:
                    for t_idx in range(n_trials):
                        self.trial_map.append((f_idx, t_idx))

                global_offset += n_trials
            except Exception as e:
                print(f"[Warning] Error reading {path}: {e}")

    @staticmethod
    def split_trials(config, filepath, val_size=0.2, test_size=0.2, random_state=42):
        """trial indices를 train/val/test로 분할"""
        file_handler = NpyFileHandler()
        total_trials = 0
        for path in filepath:
            shape = file_handler.get_meta(path)
            total_trials += shape[0]

        all_indices = list(range(total_trials))

        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=test_size,
            random_state=random_state, shuffle=True
        )

        val_ratio_in_trainval = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_in_trainval,
            random_state=random_state, shuffle=True
        )

        print(f"[*] Trial Split: Total={total_trials}, "
              f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        return train_indices, val_indices, test_indices



def get_patch_time_ms_stride(patch_idx, time_bin, stride, sampling_rate, start_time_ms=-200):
    """
    stride를 고려한 패치 인덱스 -> 실제 시간(ms) 변환.
    기존 get_patch_time_ms는 stride=time_bin을 가정하지만,
    이 함수는 stride가 time_bin과 다를 때도 올바르게 계산.
    """
    stride_ms = stride / sampling_rate * 1000
    window_ms = time_bin / sampling_rate * 1000
    start_ms = patch_idx * stride_ms + start_time_ms
    end_ms = start_ms + window_ms
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



def discover_all_subjects(data_dir):
    """
    processed_train/npy와 processed_test/npy에서
    모든 피험자 파일을 찾아 {subject_id: [file_paths]} 형태로 반환.
    """
    train_dir = os.path.join(data_dir, "processed_train/npy")
    test_dir = os.path.join(data_dir, "processed_test/npy")

    subject_files = {}

    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if (f.endswith('.npy')
                and 'label' not in f
                and 'stats' not in f
                and 'info' not in f):
                subject_id = f.replace('.npy', '')
                filepath = os.path.join(d, f)
                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                subject_files[subject_id].append(filepath)

    return subject_files

