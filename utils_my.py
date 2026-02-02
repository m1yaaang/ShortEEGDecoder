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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from EEGNet.EEGNet_util import EEGNet 



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


class InferenceManager:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
        # 1. 내부 모델 초기화(config를 넘겨줌)
        self.predictor = Predictor(config)    # 모델 결과 추론
        self.loader = ResultLoader(config)    # CSV 로더(추론 대신)
        self.evaluator = Evaluator(config)    # Metric 평가
        self.recorder = Recorder(config)      # CSV 저장
        self.visualizer = Visualizer(config)  # 시각화
        
        # 2. 체크포인트 경로 자동 탐색(config_dir)
        self.ckpt_paths = self._discover_checkpoints()
        
        # 3. Patch 개수 자동 계산
        self.num_patches = self._calculate_num_patches()

    def _discover_checkpoints(self):
        """ckpt_dir 내의 모든 .pth파일을 재귀적으로 찾거나 구조에 맞춰 탐색"""
        if not os.path.exists(self.config['ckpt_dir']):
            print(f" Checkpoint dir not found: {self.config['ckpt_dir']}")
            return []
            
        # 예: patch_0, patch_1 폴더 내부의 .pth 파일등 중 best만 찾거나 전체 리스트 업
        # 사용자 파일 구조에 맞춰 패턴 사용
        best_model_paths = []

        ckpt_lists = os.listdir(self.config['ckpt_dir'])
        # ckpt_lists = [d for d in ckpt_lists if os.path.exists(os.path.join(self.config['ckpt_dir'], d))] # 폴더 안에 logs 안에 ckpt가 있는 경우
        ckpt_lists.sort()

        """
        patch 0
            - logs
                - ckpt1
                - ckpt2
        patch 1
            - logs
        """
        for p in ckpt_lists:
            if "None" in p:
                continue
            ckpt_best_acc = -1.0
            ckpt_best_file = None
            patch_dir = os.path.join(self.config['ckpt_dir'], p)
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
        
        # best는 여기서 구현
        # epoch으로 고정하던지 best만 선택하는 함수를 추가 구현하기
        return best_model_paths

    def _calculate_num_patches(self):
        """데이터 전체 길이를 time_bin으로 나누어 총 패치 수 계산"""
        # 임시로 데이터 하나를 로드해서 전체 길이를 파악
        temp_ds = COMBDataset(config=self.config, filepath=self.config["test_files"])
        total_len, _ = temp_ds._get_sample_info()
        return total_len // self.config['time_bin']


    def run(self):
        print(f" Start Pipeline in Skip Prediction[{self.config['skip_pred']}]")
        print(f"   - Found {len(self.ckpt_paths)} checkpoints")
        print(f"   - Total Test Patches: {self.num_patches}")

        global_tgm_data = []

        for ckpt in self.ckpt_paths:
            print(f"\n==> Evaluating Checkpoint: {ckpt}")
            train_patch_idx = self._parse_patch_from_path(ckpt)
            train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
            print(f"Train Patch{train_patch_idx}({train_start_ms}~{train_end_ms})")  

            save_dir = os.path.dirname(ckpt).replace("checkpoints", "analysis")
            # net = None
            model = self.model
            if not self.config["skip_pred"]:
                model = self._load_model_instance(ckpt, self.model)

            for test_patch_idx in range(self.num_patches):
                # Test Loader 생성
                test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
                print(f"Test Patch{test_patch_idx}({test_start_ms}~{test_end_ms})")
                
                #[Step 1] Data source
                if not self.config["skip_pred"]:
                    preds,labels,probs = self.predictor.predict(model, test_patch_idx)
                    self.recorder.save_detail_csv(save_dir, train_patch_idx, test_patch_idx, preds, labels, probs)
                else: #Only Analysis
                    preds,labels,probs = self.loader.load_csv(train_patch_idx, test_patch_idx)

                #[Step 2] Evaluation
                metrics  = self.evaluator.compute_metrics(preds, labels, probs, params=self.config["metrics"])

                #[Step 3] Visualization (CM)
                self.visualizer.plot_cm(save_dir, labels, preds, train_patch_idx, test_patch_idx, metrics['acc'])

                #[Step 4] TGM Data Collection
                global_tgm_data.append({
                    'train_patch_idx': train_patch_idx,
                    'train_time_ms': train_start_ms,
                    'test_patch_idx': test_patch_idx,
                    'test_time_ms': test_start_ms,
                    'test_acc': metrics['acc'],
                    'test_bal_acc': metrics['bal_acc']
                })
            
        if model: del model
        torch.cuda.empty_cache()

        # [Step 5] Final TGM Plotting 
        if global_tgm_data:
            df_summary = pd.DataFrame(global_tgm_data)
            self.recorder.save_summary_csv(df_summary, save_dir=save_dir)

            # TGM 시각화 (Acc, Bal_Acc)
            self.visualizer.plot_tgm(ckpt, df_summary, metric_key='test_acc', save_dir=save_dir)
            self.visualizer.plot_tgm(ckpt, df_summary, metric_key='test_bal_acc', save_dir=save_dir)

        return

    # --- Helper Methods ---
    def _parse_patch_from_path(self, ckpt_path):
        # 파일명에서 patch_idx 추출 로직(사용자 규칙에 맞게)
        # 예: ".../patch_3/..." -> 3
        try:
            return int(ckpt_path.split("patch_")[1].split("/")[0])
        except:
            return 0 # 파싱 실패시 예외 처리

    def _load_model_instance(self, ckpt_path, net):
        # Config에 있는 model로 인스턴스 생성후 가중치 로드
        
        # model = self.net(n_channels=input_ch, n_timepoints=input_len, n_classes=self.config['n_classes'])
        
        checkpoint = torch.load(ckpt_path)
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        net.load_state_dict(state_dict = state)
        net.cuda().eval()
        return net
    

    def _create_loader(self, patch_idx):
        pass

class Predictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, model, test_patch_idx):

        test_config = self.config.copy()
        test_config["patch_idx"] = test_patch_idx

        test_patch_dataset = COMBDataset(config=test_config, filepath=test_config["test_files"])
        test_patch_loader = torch.utils.data.DataLoader(
                                            test_patch_dataset,
                                            batch_size=self.config["batch_size"],
                                            shuffle=False,
                                            num_workers=self.config["num_workers"],
                                            collate_fn=torch_collate_fn)


        model.eval() 

        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels, _, _ in test_patch_loader:
                inputs = inputs.cuda(0)
                outputs = model(inputs)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    logits = outputs[-1]  # Get the last output as Logits (h)
                else:
                    logits = outputs
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        return all_preds, all_labels, all_probs

class ResultLoader:
    def __init__(self, config):
        self.csv_dir = config['csv_input_dir']

    def load_csv(self, train_idx, test_idx):
        preds, labels, probs = [], [], []
        csv_path = os.path.join(self.csv_dir, f"TrainP{train_idx}_TestP{test_idx}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            preds = df['prediction'].values
            labels = df['label'].values
            probs = df['probability'].values
        return preds, labels, probs

class Evaluator:
    def __init__(self, config):
        self.is_label_null = config["is_label_null"]
    def compute_metrics(self, preds, labels, probs,params):
        results = {}  
        
        for param in params:
            if param == 'acc':
                if self.is_label_null: 
                    results['acc'] = self._acc_w_null(preds, labels, metrics='acc')
                else:
                    results['acc'] = accuracy_score(labels, preds)
            
            if param == 'bal_acc':
                if self.is_label_null:
                    results['bal_acc'] = self._acc_w_null(preds, labels, metrics='bal_acc')
                else:
                    results['bal_acc'] = balanced_accuracy_score(labels, preds)
            
            if param == "auc":
                # Multi-class AUC (one-vs-rest)
                try:
                    results['auc'] = roc_auc_score(labels, probs, multi_class='ovr')
                except:
                    results['auc'] = 0.0
            
            if param == "recall":
                results['recall'] = recall_score(labels, preds, average='macro')
            
            if param == "precision":
                results['precision'] = precision_score(labels, preds, average='macro', zero_division=0.0)
            
            if param == "fmeasure":
                precision = precision_score(labels, preds, average='macro', zero_division=0.0, labels=np.unique(labels))
                recall = recall_score(labels, preds, average='macro', zero_division=0.0, labels=np.unique(labels))
                
                if precision + recall > 0:
                    results['fmeasure'] = 2 * precision * recall / (precision + recall)
                else:
                    results['fmeasure'] = 0.0
        return results
    def _acc_w_null(self, preds, labels, metrics):

        np_preds = np.array(preds)
        np_labels = np.array(labels)

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

        if metrics == 'acc':
            return acc
        elif metrics == 'bal_acc':
            return bal_acc
        else:
            raise ValueError(f"Unsupported metric: {metrics}")


class Recorder:
    def __init__(self, config):
        self.time_bin = config["time_bin"]
        self.sr = config["sampling_rate"]
        self.save_dir = config["save_dir"]

    def save_detail_csv(self, save_dir, train_patch_idx, test_patch_idx, preds, labels, probs):
        csv_dir = os.path.join(save_dir,'csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_results.csv"
        csv_save_path = os.path.join(csv_dir, csv_filename)


        train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.time_bin, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.time_bin, self.sr)
        results_detail = {
            'train_patch_idx': train_patch_idx,
            'train_time_ms': train_start_ms,
            'test_patch_idx': test_patch_idx,
            'test_time_ms': test_start_ms,
            'prediction': preds,
            'labels': labels,
        }
        if len(probs.shape) > 1: 
            for i in range(probs.shape[1]):
                results_detail[f"prob_{i}"] = probs[:, i] 

        df_detail = pd.DataFrame(results_detail)
        df_detail.to_csv(csv_save_path, index=False)
    
    def save_summary_csv(self, df, save_dir=None):
        if save_dir == None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            raise ValueError(f"There's no {save_dir}")

        summary_csv_path = os.path.join(save_dir, "summary.csv")
        df.to_csv(summary_csv_path, index=False)

class Visualizer:
    def __init__(self, config):
        self.time_bin = config["time_bin"]
        self.sr = config["sampling_rate"]

    def plot_cm(self, save_dir, labels, preds, train_patch_idx, test_patch_idx, final_acc,epoch=None):

        train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.time_bin, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.time_bin, self.sr)

        cm_dir = os.path.join(save_dir, 'cm')
        os.makedirs(cm_dir, exist_ok=True)

        # Confusion Matrix 저장
        if epoch == None: 
            cm_title = f"Train P{train_patch_idx}({train_start_ms}~{train_end_ms}) / Test P{test_patch_idx}({test_start_ms}~{test_end_ms})\nAcc: {final_acc:.2f}%"
            cm_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_Acc{final_acc:.2f}_cm.png"
        else:
            cm_title = f"Train P{train_patch_idx}({train_start_ms}~{train_end_ms}) / Test P{test_patch_idx}({test_start_ms}~{test_end_ms})Epoch{epoch}\nAcc: {final_acc:.2f}%"
            cm_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_Epoch{epoch}_Acc{final_acc:.2f}_cm.png"

        cm_save_path = os.path.join(cm_dir, cm_filename)
        label_names = np.unique(labels)

        cm = confusion_matrix(labels, preds, labels=label_names)

        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=[f"Class_{i}" for i in label_names],
                    yticklabels=[f"Class_{i}" for i in label_names])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title(cm_title)
        ax_cm.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(cm_save_path, dpi=150)
        plt.close('all')
        return fig_cm


    def plot_tgm(self, ckpt, df, metric_key='test_acc', save_dir=None):
        """
        [기능]
        - DataFrame의 모든 숫자 데이터(idx, ms)를 이용해 TGM Heatmap을 그립니다.
        - 문자열 컬럼(_str)이 없어도 동작하며, 모든 라벨을 전부 표시합니다.
        """
        if save_dir == None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            raise ValueError(f"There's no {save_dir}")
        # 1. Pivot Table (Matrix 변환)
        tgm_matrix = df.pivot(index='train_patch_idx', columns='test_patch_idx', values=metric_key)
        matrix_values = tgm_matrix.values
        tgm_matrix.sort_index(ascending=False, inplace=True)
        
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



# # # data inference check!
# if __name__ == "__main__":
#     config = {
#                 "data_dir": "./EEG(500Hz)_COMB/processed_test/npy",
#                 "batch_size": 4,
#                 "num_workers": 0,
#                 "shuffle": True,
#                 "sampling_rate": 500,
#                 "start_time_ms" : -200,
#                 "data_ext": "npy",
#                 "window_type": "fixed",  # "fixed" or "random"
#                 "time_bin": 32,
#                 "file_chunk_type": "subject", # "subject" or "run"
#                 "normalize_method": "zscore", # "zscore" or "minmax"
#                 "patch_idx": None,
#                 "stride": None,
#                 "save_dir": "EEGNet/logs",
#                 "num_epochs": 100,
#                 "patience": 10,
#                 "n_classes": 6,
#                 "is_label_null": True,
#                 "skip_pred" : False,
#                 "metrics":["acc", "bal_acc"],
#                 "ckpt_dir":"./EEGNet/logs",
#                 "csv_input_dir":None,


#     }

#     train_files = ['./EEG(500Hz)_COMB/processed_train/npy/sub-14.npy']

#     input_len, input_ch = COMBDataset(config=config, filepath = train_files)._get_sample_info()
#     # # 모델 초기화 (53채널, 448 timepoints, 5 classes)
#     net = EEGNet(n_channels=input_ch, n_timepoints=input_len, n_classes=config["n_classes"]).cuda(0)
#     print(f"FC input size: {net.fc_input_size}")

#     best_model_paths = []
#     ckpt_lists = os.listdir(config["ckpt_dir"])
#     ckpt_lists = [d for d in ckpt_lists if os.path.isdir(os.path.join(config["ckpt_dir"], d))]
#     ckpt_lists.sort()

#     for p in ckpt_lists:
#         if "None" in p:
#             continue
#         ckpt_best_acc = -1.0
#         ckpt_best_file = None
#         patch_dir = os.path.join(config["ckpt_dir"], p)
#         ckpts = [f for f in os.listdir(patch_dir) if f.endswith(".pth")]
#         for c in ckpts:
#             try:
#                 c_acc = float(c.split("_acc=")[-1].replace(".pth",""))
#                 if c_acc > ckpt_best_acc:
#                     ckpt_best_acc = c_acc
#                     ckpt_best_file = c
#             except:
#                 continue
#         best_model_paths.append(os.path.join(patch_dir, ckpt_best_file))
        
#     # test_config = config.copy()
#     # test_config["data_dir"] = "./EEG(500Hz)_COMB/processed_test/npy"

#     manager = InferenceManager(config = config, model = net)
#     manager.run()
