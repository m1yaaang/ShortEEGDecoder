# `python EEGPT/run_EEGPT_within_inference.py`
import sys
import os
import torch
from datetime import datetime
from EEGPT.finetune_EEGPT_combine_LoRA_conv_within_util import LitEEGPTCausal_LoRA, WithinSubjectDataset
from utils_my import InferenceManager, COMBDataset, torch_collate_fn, get_patch_time_ms, Recorder, Visualizer
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class WithinRecorder(Recorder):
    """ckpt 파일명을 포함한 저장을 위한 Recorder"""

    def save_detail_csv(self, save_dir, train_patch_idx, test_patch_idx, preds, labels, probs, ckpt_name=None):
        csv_dir = os.path.join(save_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        # ckpt_name이 있으면 파일명에 포함
        if ckpt_name:
            csv_filename = f"{ckpt_name}_TestP{test_patch_idx}_results.csv"
        else:
            csv_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_results.csv"

        csv_save_path = os.path.join(csv_dir, csv_filename)

        train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.time_bin, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.time_bin, self.sr)

        results_detail = {
            'ckpt_name': ckpt_name if ckpt_name else f"TrainP{train_patch_idx}",
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

    def save_summary_csv(self, df, save_dir=None, ckpt_name=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if ckpt_name:
            summary_csv_path = os.path.join(save_dir, f"{ckpt_name}_summary.csv")
        else:
            summary_csv_path = os.path.join(save_dir, "summary.csv")

        df.to_csv(summary_csv_path, index=False)


class WithinVisualizer(Visualizer):
    """ckpt 파일명을 포함한 시각화를 위한 Visualizer"""

    def plot_cm(self, save_dir, labels, preds, train_patch_idx, test_patch_idx, final_acc, epoch=None, ckpt_name=None):
        train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.time_bin, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.time_bin, self.sr)

        cm_dir = os.path.join(save_dir, 'cm')
        os.makedirs(cm_dir, exist_ok=True)

        # ckpt_name 포함 파일명
        if ckpt_name:
            cm_title = f"{ckpt_name}\nTest P{test_patch_idx}({test_start_ms}~{test_end_ms})\nAcc: {final_acc:.2f}%"
            cm_filename = f"{ckpt_name}_TestP{test_patch_idx}_Acc{final_acc:.2f}_cm.png"
        else:
            cm_title = f"Train P{train_patch_idx}({train_start_ms}~{train_end_ms}) / Test P{test_patch_idx}({test_start_ms}~{test_end_ms})\nAcc: {final_acc:.2f}%"
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

    def plot_tgm(self, ckpt, df, metric_key='test_acc', save_dir=None, ckpt_name=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # pivot_table: 중복 (train_patch, test_patch) 조합은 평균으로 집계
        tgm_matrix = df.pivot_table(index='train_patch_idx', columns='test_patch_idx', values=metric_key, aggfunc='mean')
        tgm_matrix.sort_index(ascending=False, inplace=True)
        matrix_values = tgm_matrix.values

        train_indices = tgm_matrix.index.tolist()
        test_indices = tgm_matrix.columns.tolist()

        if 'train_time_ms' in df.columns:
            train_map = df[['train_patch_idx', 'train_time_ms']].drop_duplicates().set_index('train_patch_idx')
            train_labels = [f"P{idx} ({train_map.loc[idx, 'train_time_ms']:.0f}ms)" for idx in train_indices]
        else:
            train_labels = [f"P{idx}" for idx in train_indices]

        if 'test_time_ms' in df.columns:
            test_map = df[['test_patch_idx', 'test_time_ms']].drop_duplicates().set_index('test_patch_idx')
            test_labels = [f"P{idx} ({test_map.loc[idx, 'test_time_ms']:.0f}ms)" for idx in test_indices]
        else:
            test_labels = [f"P{idx}" for idx in test_indices]

        plt.figure(figsize=(15, 13))

        ax = sns.heatmap(
            matrix_values,
            annot=False,
            fmt=".1f",
            cmap="viridis",
            xticklabels=test_labels,
            yticklabels=train_labels,
            cbar_kws={"label": metric_key}
        )

        title = f"TGM ({metric_key})"
        if ckpt_name:
            title = f"{ckpt_name}\n{title}"
        plt.title(title, fontsize=18, pad=20)
        plt.xlabel("Test Time", fontsize=14)
        plt.ylabel("Train Time", fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.plot([0, len(test_labels)], [len(train_labels),0],
                color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        plt.tight_layout()

        if ckpt_name:
            save_path = os.path.join(save_dir, f"{ckpt_name}_TGM_{metric_key}.png")
        else:
            save_path = os.path.join(save_dir, f"TGM_{metric_key}_full.png")

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[*] TGM Heatmap saved at: {save_path}")


class EEGPTWithinInference(InferenceManager):
    def __init__(self, config, model, test_trial_indices=None):
        self.test_trial_indices = test_trial_indices
        super().__init__(config, model)
        # Override recorder와 visualizer를 Within 버전으로 교체
        self.recorder = WithinRecorder(config)
        self.visualizer = WithinVisualizer(config)

    def _discover_checkpoints(self):

        root_ckpt_dir = self.config["ckpt_dir"]

        if not os.path.exists(root_ckpt_dir):
            print(f" Checkpoint dir not found: {root_ckpt_dir}")
            return []

        # Within-Subject 모델만 필터링
        all_dirs = os.listdir(root_ckpt_dir)
        model_ckpt_list = []
        for m in all_dirs:
            # if '20260203_1912_' in m or '20260203_1913_' in m:
            #     model_ckpt_list.append(m)
            model_ckpt_list.append(m)
 
        model_ckpt_list.sort()

        best_model_paths = []

        for model_ckpt in model_ckpt_list:

            ckpt_path = os.path.join(root_ckpt_dir, model_ckpt, "checkpoints")
            
            if not os.path.exists(ckpt_path):
                print(f"  [Skip] No checkpoints dir found: {ckpt_path}")
                continue

            ckpt_lists = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt") and f!="last.ckpt"]
            ckpt_lists.sort()

            min_loss = float('inf')
            best_file = None

            # for c in ckpt_lists:
            #     try:
            #         if "loss=" in c:
            #             loss_str = c.split("loss=")[-1].replace(".ckpt", "")
            #             c_loss = float(loss_str)

            #             if c_loss < min_loss:
            #                 min_loss = c_loss
            #                 best_file = c
            #     except Exception as e:
            #         print(f"  [Warning] Parsing failed for {c}: {e}")
            #         continue
            # if best_file:
            #     full_path = os.path.join(ckpt_path, best_file)
            #     best_model_paths.append(full_path)
            # else:
            #     print(f"  [Skip] No valid checkpoint file found in {ckpt_path}")

            full_path = [os.path.join(ckpt_path, c) for c in ckpt_lists]
            best_model_paths.extend(full_path)

        return best_model_paths


    def _parse_patch_from_path(self, filepath):
        match = re.search(r"_P(\d+)_", filepath)
        if match:
            return int(match.group(1))
        print(f"[Warning] Failed to extract patch number from: {filepath}")
        return 0

    def _load_model_instance(self, ckpt_path, net):
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint

        msg = net.load_state_dict(state, strict=False)
        print(f"[*] Model Loaded: {msg}")

        net.cuda().eval()
        return net

    def predict_within_subject(self, model, test_patch_idx):
        """
        Within-Subject용 predict: test_trial_indices를 사용하여 테스트
        """
        test_config = self.config.copy()
        test_config["patch_idx"] = test_patch_idx

        # WithinSubjectDataset 사용 (test_trial_indices로 필터링)
        test_dataset = WithinSubjectDataset(
            config=test_config,
            filepath=self.config["test_files"],
            trial_indices=self.test_trial_indices
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=torch_collate_fn
        )

        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, _, mask in test_loader:
                inputs = inputs.cuda(0)
                outputs = model(inputs, mask=mask)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    logits = outputs[-1]
                else:
                    logits = outputs
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def _get_ckpt_name(self, ckpt_path):
        """ckpt 파일명에서 저장용 이름 추출 (.ckpt 제거)"""
        return os.path.basename(ckpt_path).replace(".ckpt", "")

    def run(self):
        """Within-Subject용 run 메서드"""
        print(f" Start Within-Subject Inference")
        print(f"   - Found {len(self.ckpt_paths)} checkpoints")
        print(f"   - Test Trials: {len(self.test_trial_indices) if self.test_trial_indices else 'All'}")
        print(f"   - Total Test Patches: {self.num_patches}")

        global_tgm_data = []

        for ckpt in self.ckpt_paths:
            print(f"\n==> Evaluating Checkpoint: {ckpt}")
            ckpt_name = self._get_ckpt_name(ckpt)
            train_patch_idx = self._parse_patch_from_path(ckpt)
            train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
            print(f"Checkpoint: {ckpt_name}")
            print(f"Train Patch{train_patch_idx}({train_start_ms}~{train_end_ms})")

            save_dir = os.path.dirname(ckpt).replace("checkpoints", "analysis")
            model = self.model
            if not self.config["skip_pred"]:
                model = self._load_model_instance(ckpt, self.model)

            ckpt_tgm_data = []  # 각 ckpt별 TGM 데이터

            for test_patch_idx in range(self.num_patches):
                test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
                print(f"Test Patch{test_patch_idx}({test_start_ms}~{test_end_ms})")

                if not self.config["skip_pred"]:
                    preds, labels, probs = self.predict_within_subject(model, test_patch_idx)
                    self.recorder.save_detail_csv(save_dir, train_patch_idx, test_patch_idx, preds, labels, probs, ckpt_name=ckpt_name)
                else:
                    preds, labels, probs = self.loader.load_csv(train_patch_idx, test_patch_idx)

                metrics = self.evaluator.compute_metrics(preds, labels, probs, params=self.config["metrics"])
                self.visualizer.plot_cm(save_dir, labels, preds, train_patch_idx, test_patch_idx, metrics['acc'], ckpt_name=ckpt_name)

                ckpt_tgm_data.append({
                    'ckpt_name': ckpt_name,
                    'train_patch_idx': train_patch_idx,
                    'train_time_ms': train_start_ms,
                    'test_patch_idx': test_patch_idx,
                    'test_time_ms': test_start_ms,
                    'test_acc': metrics['acc'],
                    'test_bal_acc': metrics['bal_acc']
                })
                global_tgm_data.append(ckpt_tgm_data[-1])

            # 각 ckpt별 summary 및 TGM 저장
            if ckpt_tgm_data:
                df_ckpt_summary = pd.DataFrame(ckpt_tgm_data)
                self.recorder.save_summary_csv(df_ckpt_summary, save_dir=save_dir, ckpt_name=ckpt_name)
                self.visualizer.plot_tgm(ckpt, df_ckpt_summary, metric_key='test_acc', save_dir=save_dir, ckpt_name=ckpt_name)
                self.visualizer.plot_tgm(ckpt, df_ckpt_summary, metric_key='test_bal_acc', save_dir=save_dir, ckpt_name=ckpt_name)

        if model: del model
        torch.cuda.empty_cache()

        # -------------------------------------------------------
        # 전체 TGM 생성 (모든 epoch/patch 결과 통합)
        # -------------------------------------------------------
        if global_tgm_data:
            df_summary = pd.DataFrame(global_tgm_data)

            # 새 폴더 생성 (combined_analysis)
            combined_dir = os.path.join(self.config["ckpt_dir"], "combined_analysis")
            os.makedirs(combined_dir, exist_ok=True)

            # 전체 summary CSV 저장
            self.recorder.save_summary_csv(df_summary, save_dir=combined_dir, ckpt_name="all_epochs_combined")

            # 전체 TGM 시각화 (train_patch x test_patch 매트릭스)
            print(f"\n[*] Generating Combined TGM from all epochs...")
            self.visualizer.plot_tgm(
                ckpt="combined",
                df=df_summary,
                metric_key='test_acc',
                save_dir=combined_dir,
                ckpt_name="all_epochs_combined"
            )
            self.visualizer.plot_tgm(
                ckpt="combined",
                df=df_summary,
                metric_key='test_bal_acc',
                save_dir=combined_dir,
                ckpt_name="all_epochs_combined"
            )

            print(f"[*] Combined TGM saved at: {combined_dir}")

        return


# [1] 디버거가 붙어있는지 확인하는 함수
def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None

if __name__ == "__main__":



    subject_id = "sub-46"

    # [2] 상황에 따른 전략 설정
    if is_debugging():
        print("[Debug Mode Detected] Switching to Single GPU")
        num_workers = 1
    else:
        print("[Inference Mode]")
        num_workers = 16

    test_config = {
                "data_dir": "./EEG(256Hz)_COMB/",
                "batch_size": 64,
                "num_workers": num_workers,
                "shuffle": False,
                "sampling_rate": 256,
                "start_time_ms" : -200,
                "data_ext": "npy",
                "window_type": "fixed",
                "time_bin": 8,
                "file_chunk_type": "subject",
                "normalize_method": "zscore",
                "patch_idx": None,
                "stride": None,
                "save_dir": "EEGPT/within/T2_256Hz_t8",
                "num_epochs": 100,
                "patience": 10,
                "n_classes": 6,
                "is_label_null": True,
                "skip_pred" : False,
                "metrics":["acc", "bal_acc"],
                "ckpt_dir":"./EEGPT/within/T2_256Hz_t8",
                "csv_input_dir":None,
    }

    # Within-Subject: sub-46 파일만 사용
    train_dir = os.path.join(test_config["data_dir"], "processed_train/npy")
    test_files = [
            os.path.join(train_dir, f) for f in os.listdir(train_dir)
            if "label" not in f and "stats" not in f and "info" not in f and subject_id in f
        ]

    test_config["test_files"] = test_files

    # Trial Split (학습 때와 동일한 random_state 사용!)
    train_trial_indices, val_trial_indices, test_trial_indices = WithinSubjectDataset.split_trials(
        config=test_config,
        filepath=test_files,
        val_size=0.2,
        test_size=0.2,
        random_state=42  # 학습 때와 동일!
    )

    print(f"\n[*] Using Test Trial Indices: {len(test_trial_indices)} trials")

    input_len, input_ch = COMBDataset(config=test_config, filepath=test_files)._get_sample_info()
    n_patches = input_len // test_config["time_bin"]

    test_config["n_patches"] = n_patches
    test_config["input_len"] = input_len
    test_config["input_ch"] = input_ch

    # Init model
    model = LitEEGPTCausal_LoRA(
        config=test_config,
        fixed_train_patch_idx=0,
    )

    manager = EEGPTWithinInference(test_config, model=model, test_trial_indices=test_trial_indices)
    manager.run()

    del model, manager
    torch.cuda.empty_cache()
