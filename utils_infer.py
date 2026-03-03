import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, recall_score, precision_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns   
from utils_data import (discover_all_subjects, WithinSubjectDatasetTotal,
                        get_patch_time_ms_stride, COMBDataset)
from tqdm import tqdm

class InferenceManager:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # Config-derived attributes
        self.time_bin = config["time_bin"]
        self.stride = config.get("stride") or config["time_bin"]

        # 1. 내부 모델 초기화(config를 넘겨줌)
        self.predictor = Predictor(config)    # 모델 결과 추론
        self.loader = ResultLoader(config)    # CSV 로더(추론 대신)
        self.evaluator = Evaluator(config)    # Metric 평가
        self.recorder = Recorder(config)      # CSV 저장
        self.visualizer = Visualizer(config)  # 시각화

        # 2. 모델 signature 캐싱 (매번 inspect 호출 방지)
        import inspect
        sig = inspect.signature(model.forward)
        self._use_mask = 'mask' in sig.parameters
        self._use_patch_size = 'patch_size' in sig.parameters
        

    def _discover_subject_checkpoints(self, subject_dir):
        """한 피험자의 모든 checkpoint를 탐색"""
        ckpt_paths = []
        if not os.path.exists(subject_dir):
            return ckpt_paths

        for exp_name in sorted(os.listdir(subject_dir)):
            ckpt_dir = os.path.join(subject_dir, exp_name, "checkpoints")
            if not os.path.exists(ckpt_dir):
                continue
            ckpts = [f for f in os.listdir(ckpt_dir)
                     if f.endswith(".ckpt") and f != "last.ckpt"]
            for c in sorted(ckpts):
                ckpt_paths.append(os.path.join(ckpt_dir, c))

        return ckpt_paths

    def _calculate_num_patches(self, input_len):
        """stride를 고려한 패치 수 계산"""
        return (input_len - self.time_bin) // self.stride + 1
    
    def _parse_patch_from_path(self, filepath):
        match = re.search(r"_P(\d+)_", filepath)
        if match:
            return int(match.group(1))
        print(f"[Warning] Failed to extract patch number from: {filepath}")
        return 0


    def _parse_epoch_from_ckpt(self, ckpt_name):
        match = re.search(r'epoch=(\d+)', ckpt_name)
        return int(match.group(1)) if match else -1

    def _get_ckpt_name(self, ckpt_path):
        return os.path.basename(ckpt_path).replace(".ckpt", "")

    def _load_model_instance(self, ckpt_path, net):
        """Load checkpoint into model (compatible with EEGNet, EEGPT, Lightning)"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Extract state_dict from various checkpoint formats
        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint

        # Check if keys need prefix stripping (Lightning wraps model as self.model)
        sample_key = next(iter(state.keys()), '')
        net_keys = set(net.state_dict().keys())

        # If state keys have 'model.' prefix but net doesn't expect it, strip prefix
        if sample_key.startswith('model.') and not any(k.startswith('model.') for k in net_keys):
            state = {k[6:]: v for k, v in state.items() if k.startswith('model.')}

        msg = net.load_state_dict(state, strict=False) 
        print(f"[*] Model Loaded: {ckpt_path} {msg}")

        net.cuda().eval()
        return net

    def _create_dataset(self, subject_files, test_trial_indices,
                        patch_idx, shared_cache):
        """캐시를 공유하는 Dataset 생성 (DataLoader 없이 직접 접근)"""
        test_config = self.config.copy()
        test_config["patch_idx"] = patch_idx
        return WithinSubjectDatasetTotal(
            config=test_config,
            filepath=subject_files,
            trial_indices=test_trial_indices,
            shared_cache=shared_cache,
        )

    def predict_from_dataset(self, model, dataset):
        """DataLoader 없이 Dataset에서 직접 배치 추론 (EEGNet/EEGPT 호환, FP16 최적화)"""
        model.eval()
        batch_size = self.config["batch_size"]
        n = len(dataset)
        patch_size = self.config.get("patch_size")

        # 전체 데이터를 한 번에 수집 (캐싱된 mmap → 빠름)
        all_X, all_Y, all_mask = [], [], []
        for i in range(n):
            X, Y, stat, mask = dataset[i]
            all_X.append(X)
            all_Y.append(Y)
            all_mask.append(mask)

        X_t = torch.cat(all_X, dim=0)       # (N, Ch, T)
        Y_t = torch.cat(all_Y, dim=0)       # (N,)
        mask_t = torch.cat(all_mask, dim=0)  # (N, Ch, T)

        all_preds, all_probs = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():  # FP16 autocast
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                x_b = X_t[s:e].cuda(0)

                # Build kwargs based on cached model signature
                kwargs = {}
                if self._use_mask:
                    kwargs['mask'] = mask_t[s:e].cuda(0)
                if self._use_patch_size and patch_size is not None:
                    kwargs['patch_size'] = patch_size

                outputs = model(x_b, **kwargs)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                probs = F.softmax(logits.float(), dim=1)  # float for softmax precision
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        return (torch.cat(all_preds).numpy(),
                Y_t.numpy(),
                torch.cat(all_probs).numpy())

    def _discover_trained_subjects(self):
        """ckpt_dir 내에서 checkpoint가 존재하는 피험자만 탐색"""
        root_dir = self.config["ckpt_dir"]
        trained = {}
        if not os.path.exists(root_dir):
            return trained
        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path) or not entry.startswith("sub-"):
                continue
            ckpts = self._discover_subject_checkpoints(entry_path)
            if ckpts:
                trained[entry] = ckpts
        return trained

    def run(self):
        """Run inference across trained subjects only, generate cross-subject epoch TGMs"""
        subject_files_map = discover_all_subjects(self.config["data_dir"])
        trained_subjects = self._discover_trained_subjects()

        subject_ids = sorted(trained_subjects.keys())
        print(f"[*] Found {len(subject_ids)} trained subjects for inference")

        root_dir = self.config["ckpt_dir"]
        global_all_data = []

        subj_pbar = tqdm(subject_ids, desc="Subjects", unit="subj")
        for subject_id in subj_pbar:
            subj_pbar.set_description(f"Subject {subject_id}")

            if subject_id not in subject_files_map:
                print(f"  [Skip] No data files found for {subject_id}")
                continue

            subj_files = subject_files_map[subject_id]
            ckpt_paths = trained_subjects[subject_id]

            # Trial split (same random_state as training)
            _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
                config=self.config, filepath=subj_files,
                val_size=0.2, test_size=0.2, random_state=42
            )

            # Calculate n_patches
            input_len, input_ch = COMBDataset(
                config=self.config, filepath=subj_files
            )._get_sample_info()
            num_patches = self._calculate_num_patches(input_len)

            print(f"  Checkpoints: {len(ckpt_paths)}")
            print(f"  Test Trials: {len(test_trial_indices)}")
            print(f"  Patches: {num_patches} (stride={self.stride})")

            # Subject 단위 shared cache (mmap 참조 재사용)
            shared_cache = {}

            # Patch별 Dataset 미리 생성 (캐시 공유, DataLoader 없음)
            patch_datasets = {}
            for p_idx in range(num_patches):
                patch_datasets[p_idx] = self._create_dataset(
                    subj_files, test_trial_indices, p_idx, shared_cache)

            subject_data = []

            ckpt_pbar = tqdm(ckpt_paths, desc="  Checkpoints", unit="ckpt", leave=False)
            for ckpt_path in ckpt_pbar:
                ckpt_name = self._get_ckpt_name(ckpt_path)
                train_patch_idx = self._parse_patch_from_path(ckpt_path)
                epoch = self._parse_epoch_from_ckpt(ckpt_name)
                train_start_ms, _ = get_patch_time_ms_stride(
                    train_patch_idx, self.time_bin, self.stride,
                    self.config['sampling_rate'])

                ckpt_pbar.set_description(f"  P{train_patch_idx} ep{epoch}")

                save_dir = os.path.dirname(ckpt_path).replace(
                    "checkpoints", "analysis")

                if not self.config["skip_pred"]:
                    model = self._load_model_instance(ckpt_path, self.model)
                else:
                    model = self.model

                for test_patch_idx in tqdm(range(num_patches),
                                          desc="    TestPatch", unit="p", leave=False):
                    test_start_ms, _ = get_patch_time_ms_stride(
                        test_patch_idx, self.time_bin, self.stride,
                        self.config['sampling_rate'])

                    if not self.config["skip_pred"]:
                        preds, labels, probs = self.predict_from_dataset(
                            model, patch_datasets[test_patch_idx])

                        # self.recorder.save_detail_csv(
                        #     save_dir, train_patch_idx, test_patch_idx,
                        #     preds, labels, probs, ckpt_name=ckpt_name)
                    else:
                        # CSV loading for skip_pred mode
                        csv_dir = os.path.join(save_dir, 'csv')
                        csv_path = os.path.join(
                            csv_dir,
                            f"{ckpt_name}_TestP{test_patch_idx}_results.csv")
                        if os.path.exists(csv_path):
                            df_csv = pd.read_csv(csv_path)
                            preds = df_csv['prediction'].values
                            labels = df_csv['labels'].values
                            probs = np.zeros((len(preds), self.config["n_classes"]))
                            for i in range(self.config["n_classes"]):
                                col = f"prob_{i}"
                                if col in df_csv.columns:
                                    probs[:, i] = df_csv[col].values
                        else:
                            print(f"    [Skip] CSV not found: {csv_path}")
                            continue

                    metrics = self.evaluator.compute_metrics(
                        preds, labels, probs,
                        params=self.config["metrics"])

                    # CM은 대각선(train_patch == test_patch)만 출력
                    if train_patch_idx == test_patch_idx:
                        self.visualizer.plot_cm(
                            save_dir, labels, preds,
                            train_patch_idx, test_patch_idx,
                            metrics['acc'], ckpt_name=ckpt_name)

                    row = {
                        'subject': subject_id,
                        'ckpt_name': ckpt_name,
                        'epoch': epoch,
                        'train_patch_idx': train_patch_idx,
                        'train_time_ms': train_start_ms,
                        'test_patch_idx': test_patch_idx,
                        'test_time_ms': test_start_ms,
                        'test_acc': metrics['acc'],
                        'test_bal_acc': metrics['bal_acc'],
                    }
                    subject_data.append(row)
                    global_all_data.append(row)

            del patch_datasets, shared_cache

            # Per-subject combined summary
            if subject_data:
                df_subj = pd.DataFrame(subject_data)
                subj_analysis_dir = os.path.join(
                    root_dir, subject_id, "combined_analysis")
                os.makedirs(subj_analysis_dir, exist_ok=True)

                self.recorder.save_summary_csv(
                    df_subj, save_dir=subj_analysis_dir,
                    ckpt_name=f"{subject_id}_all")

                # Per-subject TGM (all epochs averaged)
                self.visualizer.plot_tgm(
                    ckpt=ckpt_name,
                    df=df_subj, metric_key='test_acc',
                    save_dir=subj_analysis_dir,
                    title=f"{subject_id} - All Epochs TGM (test_acc)",
                    filename_prefix=f"{subject_id}_all_epochs")
                self.visualizer.plot_tgm(
                    ckpt=ckpt_name,
                    df=df_subj, metric_key='test_bal_acc',
                    save_dir=subj_analysis_dir,
                    title=f"{subject_id} - All Epochs TGM (test_bal_acc)",
                    filename_prefix=f"{subject_id}_all_epochs")

            del model
            torch.cuda.empty_cache()

        # -------------------------------------------------------
        # Cross-subject epoch-averaged TGM
        # -------------------------------------------------------
        if not global_all_data:
            print("[!] No data collected.")
            return

        df_all = pd.DataFrame(global_all_data)
        combined_dir = os.path.join(root_dir, "cross_subject_analysis")
        os.makedirs(combined_dir, exist_ok=True)

        # Save full summary
        self.recorder.save_summary_csv(
            df_all, save_dir=combined_dir,
            ckpt_name="all_subjects_all_epochs")

        # Per-epoch cross-subject averaged TGM
        epochs = sorted(df_all['epoch'].unique())
        print(f"\n[*] Generating cross-subject epoch-averaged TGMs "
              f"for {len(epochs)} epochs...")

        for epoch_num in epochs:
            df_epoch = df_all[df_all['epoch'] == epoch_num]
            epoch_label = f"epoch_{epoch_num:02d}"

            # Average across subjects per (train_patch, test_patch)
            df_avg = df_epoch.groupby(
                ['train_patch_idx', 'test_patch_idx']
            ).agg({
                'train_time_ms': 'first',
                'test_time_ms': 'first',
                'test_acc': 'mean',
                'test_bal_acc': 'mean',
            }).reset_index()

            n_subjects = df_epoch['subject'].nunique()
            print(f"  {epoch_label}: {n_subjects} subjects, "
                  f"{len(df_avg)} patch pairs")

            epoch_dir = os.path.join(combined_dir, epoch_label)
            os.makedirs(epoch_dir, exist_ok=True)

            self.recorder.save_summary_csv(
                df_avg, save_dir=epoch_dir,
                ckpt_name=f"{epoch_label}_cross_subject_avg")

            self.visualizer.plot_tgm(
                ckpt=ckpt_name,
                df=df_avg, metric_key='test_acc',
                save_dir=epoch_dir,
                title=(f"Epoch {epoch_num} - Cross-Subject Avg TGM "
                       f"(test_acc, n={n_subjects})"),
                filename_prefix=f"{epoch_label}_cross_subject_avg")

            self.visualizer.plot_tgm(
                ckpt=ckpt_name,
                df=df_avg, metric_key='test_bal_acc',
                save_dir=epoch_dir,
                title=(f"Epoch {epoch_num} - Cross-Subject Avg TGM "
                       f"(test_bal_acc, n={n_subjects})"),
                filename_prefix=f"{epoch_label}_cross_subject_avg")

        # Overall mean TGM (all epochs, all subjects)
        df_overall_avg = df_all.groupby(
            ['train_patch_idx', 'test_patch_idx']
        ).agg({
            'train_time_ms': 'first',
            'test_time_ms': 'first',
            'test_acc': 'mean',
            'test_bal_acc': 'mean',
        }).reset_index()

        n_total_subjects = df_all['subject'].nunique()

        self.visualizer.plot_tgm(
            ckpt=ckpt_name,
            df=df_overall_avg, metric_key='test_acc',
            save_dir=combined_dir,
            title=(f"All Epochs All Subjects Avg TGM "
                   f"(test_acc, n={n_total_subjects})"),
            filename_prefix="all_epochs_all_subjects_avg")

        self.visualizer.plot_tgm(
            ckpt=ckpt_name,
            df=df_overall_avg, metric_key='test_bal_acc',
            save_dir=combined_dir,
            title=(f"All Epochs All Subjects Avg TGM "
                   f"(test_bal_acc, n={n_total_subjects})"),
            filename_prefix="all_epochs_all_subjects_avg")

        print(f"\n[!] Cross-Subject Analysis Complete. Output: {combined_dir}")



    # --- Helper Methods ---
    def _parse_patch_from_path(self, ckpt_path):
        # 파일명에서 patch_idx 추출 로직(사용자 규칙에 맞게)
        # 예: ".../patch_3/..." -> 3
        try:
            return int(ckpt_path.split("patch_")[1].split("/")[0])
        except:
            return 0 # 파싱 실패시 예외 처리

    def _load_model_instance_legacy(self, ckpt_path, net):
        # Legacy method - kept for reference
        checkpoint = torch.load(ckpt_path)
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        net.load_state_dict(state)
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
            for inputs, labels, _, mask in test_patch_loader:
                inputs = inputs.cuda(0)
                outputs = model(inputs, mask=mask)
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
        self.stride = config.get("stride") or config["time_bin"]

    def save_detail_csv(self, save_dir, train_patch_idx, test_patch_idx, 
                        preds, labels, probs, ckpt_name=None):
        csv_dir = os.path.join(save_dir,'csv')
        os.makedirs(csv_dir, exist_ok=True)

        if ckpt_name:
            csv_filename = f"{ckpt_name}_TestP{test_patch_idx}_results.csv"
        else:
            csv_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_results.csv"
        
        csv_save_path = os.path.join(csv_dir, csv_filename)

        train_start_ms, train_end_ms = get_patch_time_ms_stride(train_patch_idx, self.time_bin, self.stride, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms_stride(test_patch_idx, self.time_bin, self.stride, self.sr)
        
        results_detail = {
            'ckpt_name': ckpt_name or f"TrainP{train_patch_idx}",
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
 
        pd.DataFrame(results_detail).to_csv(csv_save_path, index=False)
    
    def save_summary_csv(self, df, save_dir=None, ckpt_name =None):
        if save_dir == None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if ckpt_name:
            path = os.path.join(save_dir, f"{ckpt_name}_summary.csv")
        else:
            path = os.path.join(save_dir, "summary.csv")

        df.to_csv(path, index=False)

class Visualizer:
    def __init__(self, config):
        self.time_bin = config["time_bin"]
        self.sr = config["sampling_rate"] 
        self.stride = config.get("stride") or config["time_bin"]

    def plot_cm(self, save_dir, labels, preds, train_patch_idx, test_patch_idx, 
                final_acc,epoch=None, ckpt_name=None):

        train_start_ms, train_end_ms = get_patch_time_ms_stride(
            train_patch_idx, self.time_bin, self.stride, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms_stride(
            test_patch_idx, self.time_bin, self.stride, self.sr)

        cm_dir = os.path.join(save_dir, 'cm')
        os.makedirs(cm_dir, exist_ok=True)

        # Confusion Matrix 저장
        if ckpt_name:
            cm_title = (f"{ckpt_name}\n"
                        f"Test P{test_patch_idx}({test_start_ms:.0f}~{test_end_ms:.0f}ms)\n"
                        f"Acc: {final_acc:.2f}%")
            cm_filename = f"{ckpt_name}_TestP{test_patch_idx}_Acc{final_acc:.2f}_cm.png"
        else:
            cm_title = (f"Train P{train_patch_idx}({train_start_ms:.0f}~{train_end_ms:.0f}ms) / "
                        f"Test P{test_patch_idx}({test_start_ms:.0f}~{test_end_ms:.0f}ms)\n"
                        f"Acc: {final_acc:.2f}%")
            cm_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_Acc{final_acc:.2f}_cm.png"

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


    def plot_tgm(self, ckpt, df, metric_key='test_acc', save_dir=None,
                title=None, filename_prefix=None):
        """
        [기능]
        - DataFrame의 모든 숫자 데이터(idx, ms)를 이용해 TGM Heatmap을 그립니다.
        - 문자열 컬럼(_str)이 없어도 동작하며, 모든 라벨을 전부 표시합니다.
        pivot_table을 사용하여 중복 (train_patch, test_patch) 조합은 평균으로 집계.
        """
        if save_dir == None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1. Pivot Table (Matrix 변환)
        tgm_matrix = df.pivot_table(
            index='train_patch_idx',
            columns='test_patch_idx',
            values=metric_key,
            aggfunc='mean'
        )
        tgm_matrix.sort_index(ascending=False, inplace=True)

        train_indices = tgm_matrix.index.tolist()
        test_indices = tgm_matrix.columns.tolist()

        # 2. 라벨 생성 (데이터프레임 내 ms 정보가 있으면 활용)
        
        # (1) Train Label 생성
        if 'train_time_ms' in df.columns:
            train_map = (df[['train_patch_idx', 'train_time_ms']]
                         .drop_duplicates()
                         .set_index('train_patch_idx'))
            train_labels = [f"P{idx} ({train_map.loc[idx, 'train_time_ms']:.0f}ms)"
                            for idx in train_indices]
        else:
            train_labels = [f"P{idx}" for idx in train_indices]

        if 'test_time_ms' in df.columns:
            test_map = (df[['test_patch_idx', 'test_time_ms']]
                        .drop_duplicates()
                        .set_index('test_patch_idx'))
            test_labels = [f"P{idx} ({test_map.loc[idx, 'test_time_ms']:.0f}ms)"
                           for idx in test_indices]
        else:
            test_labels = [f"P{idx}" for idx in test_indices]

        # 3. Plotting
        plt.figure(figsize=(15, 13)) # 라벨이 많으므로 그림 크기를 넉넉하게 잡음
        
        ax = sns.heatmap(
            tgm_matrix.values,
            annot=False,            # 칸이 빽빽하면 숫자가 겹치므로 False 권장 (필요시 True)
            fmt=".1f",
            cmap="viridis",
            xticklabels=test_labels, # [수정] 모든 라벨 리스트를 직접 전달
            yticklabels=train_labels, # [수정] 모든 라벨 리스트를 직접 전달
            cbar_kws={"label": metric_key}
        )

        # 4. 축 라벨 스타일 설정  
        xticks_stride = 10
        ax.set_xticks(np.arange(len(test_labels))[::xticks_stride] + 0.5)
        ax.set_xticklabels(test_labels[::xticks_stride], rotation=45, ha='right', fontsize=9)

        yticks_stride = 10
        ax.set_yticks(np.arange(len(train_labels))[::yticks_stride] + 0.5)
        ax.set_yticklabels(train_labels[::yticks_stride], rotation=0, fontsize=9)

        plot_title = title or f"TGM ({metric_key})"
        plt.title(plot_title, fontsize=18, pad=20)
        plt.xlabel("Test Time", fontsize=14)
        plt.ylabel("Train Time", fontsize=14)

        # X축 라벨: 45도 회전, 폰트 사이즈 조절
        plt.xticks(rotation=45, ha='right', fontsize=9) 
        
        # Y축 라벨: 0도 (가로) 유지, 폰트 사이즈 조절
        plt.yticks(rotation=0, fontsize=9)

        # 5. 대각선 (Train == Test 시점)
        plt.plot([0, len(test_labels)], [len(train_labels), 0], 
                color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        plt.tight_layout()
        
        # 저장

        prefix = filename_prefix or "TGM"
        save_path = os.path.join(save_dir, f"{prefix}_{metric_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"[*] TGM Heatmap (All Labels) saved at: {save_path}")

