# python /EEGNet_inference_total.py --subject sub-22


# `python EEGPT/run_EEGPT_within_inference_total.py`
#
# Flags:
#   --subject sub-14         : 특정 피험자만 inference
#   --subject sub-14 sub-15  : 여러 피험자 inference
#   (no --subject)           : 학습된 전체 피험자 inference
#   --cross-subject-only     : inference 건너뛰고 기존 summary.csv로 cross-subject TGM만 생성

import argparse
import sys
import os
import torch
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm

from EEGPT.finetune_EEGPT_combine_LoRA_conv_within_util_total import (
    LitEEGPTCausal_LoRA, LitEEGPTCausal_LoRA_Total,
    WithinSubjectDatasetTotal,
    get_patch_time_ms_stride,
    discover_all_subjects,
)
from utils_my import (
    InferenceManager, COMBDataset, torch_collate_fn,
    Recorder, Visualizer, Evaluator,
)


# -------------------------------------------------------
# Stride-aware Recorder
# -------------------------------------------------------
class TotalWithinRecorder(Recorder):
    """stride-aware, ckpt_name-aware Recorder"""

    def __init__(self, config):
        super().__init__(config)
        self.stride = config.get("stride") or config["time_bin"]

    def save_detail_csv(self, save_dir, train_patch_idx, test_patch_idx,
                        preds, labels, probs, ckpt_name=None):
        csv_dir = os.path.join(save_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        if ckpt_name:
            csv_filename = f"{ckpt_name}_TestP{test_patch_idx}_results.csv"
        else:
            csv_filename = f"TrainP{train_patch_idx}_TestP{test_patch_idx}_results.csv"

        csv_save_path = os.path.join(csv_dir, csv_filename)

        train_start_ms, train_end_ms = get_patch_time_ms_stride(
            train_patch_idx, self.time_bin, self.stride, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms_stride(
            test_patch_idx, self.time_bin, self.stride, self.sr)

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

    def save_summary_csv(self, df, save_dir=None, ckpt_name=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if ckpt_name:
            path = os.path.join(save_dir, f"{ckpt_name}_summary.csv")
        else:
            path = os.path.join(save_dir, "summary.csv")

        df.to_csv(path, index=False)


# -------------------------------------------------------
# Stride-aware Visualizer
# -------------------------------------------------------
class TotalWithinVisualizer(Visualizer):
    """stride-aware Visualizer with flexible TGM plotting"""

    def __init__(self, config):
        super().__init__(config)
        self.stride = config.get("stride") or config["time_bin"]

    def plot_cm(self, save_dir, labels, preds, train_patch_idx, test_patch_idx,
                final_acc, epoch=None, ckpt_name=None):
        train_start_ms, train_end_ms = get_patch_time_ms_stride(
            train_patch_idx, self.time_bin, self.stride, self.sr)
        test_start_ms, test_end_ms = get_patch_time_ms_stride(
            test_patch_idx, self.time_bin, self.stride, self.sr)

        cm_dir = os.path.join(save_dir, 'cm')
        os.makedirs(cm_dir, exist_ok=True)

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

    def plot_tgm(self, df, metric_key='test_acc', save_dir=None,
                 title=None, filename_prefix=None):
        """
        TGM Heatmap 생성.
        pivot_table을 사용하여 중복 (train_patch, test_patch) 조합은 평균으로 집계.
        """
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        tgm_matrix = df.pivot_table(
            index='train_patch_idx',
            columns='test_patch_idx',
            values=metric_key,
            aggfunc='mean'
        )
        tgm_matrix.sort_index(ascending=False, inplace=True)

        train_indices = tgm_matrix.index.tolist()
        test_indices = tgm_matrix.columns.tolist()

        # Labels with stride-aware time
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

        plt.figure(figsize=(15, 13))
        ax = sns.heatmap(
            tgm_matrix.values,
            annot=False,
            fmt=".1f",
            cmap="viridis",
            xticklabels=test_labels,
            yticklabels=train_labels,
            cbar_kws={"label": metric_key},
        )

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
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        # Diagonal line (train==test time)
        plt.plot([0, len(test_labels)], [len(train_labels), 0],
                 color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        plt.tight_layout()

        prefix = filename_prefix or "TGM"
        save_path = os.path.join(save_dir, f"{prefix}_{metric_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[*] TGM saved: {save_path}")

    def plot_temporal_accuracy(self, df, save_dir=None,
                               title=None, filename_prefix=None,
                               n_classes=6):
        """
        Diagonal accuracy (train_patch == test_patch) 시간 축 플롯.
        여러 epoch가 있으면 epoch별 평균 + 전체 평균 표시.
        """
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        for metric_key in ['test_acc', 'test_bal_acc']:
            # Epoch별 평균
            df_avg = df.groupby(['train_patch_idx', 'train_time_ms']).agg({
                metric_key: 'mean',
            }).reset_index().sort_values('train_patch_idx')

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df_avg['train_time_ms'], df_avg[metric_key],
                    linewidth=1.5, color='steelblue', label='epoch avg')
            ax.axhline(y=df_avg[metric_key].mean(), color='red',
                        linestyle='--', linewidth=1, alpha=0.7,
                        label=f"mean={df_avg[metric_key].mean():.2f}%")

            chance = 100.0 / n_classes
            ax.axhline(y=chance, color='gray', linestyle=':',
                        linewidth=1, alpha=0.5, label=f"chance={chance:.1f}%")

            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel(metric_key, fontsize=12)
            ax.set_title(title or f"Diagonal {metric_key}", fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            prefix = filename_prefix or "diagonal"
            save_path = os.path.join(save_dir, f"{prefix}_{metric_key}.png")
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"[*] Temporal accuracy saved: {save_path}")


# -------------------------------------------------------
# Inference Manager for all subjects
# -------------------------------------------------------
class EEGPTWithinInferenceTotal(InferenceManager):
    """
    All-subjects within-subject inference with stride support.
    Generates per-subject CM/TGM, and cross-subject averaged TGMs from summary CSVs.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.evaluator = Evaluator(config)
        self.recorder = TotalWithinRecorder(config)
        self.visualizer = TotalWithinVisualizer(config)
        self.stride = config.get("stride") or config["time_bin"]
        self.time_bin = config["time_bin"]

    def _calculate_num_patches(self, input_len):
        return (input_len - self.time_bin) // self.stride + 1

    def _discover_subject_checkpoints(self, subject_dir):
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
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        except (RuntimeError, Exception) as e:
            print(f"[!] Corrupt checkpoint, skipping: {ckpt_path}\n    {e}")
            return None

        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint

        msg = net.load_state_dict(state, strict=False)

        net.cuda().eval()
        return net

    def _create_dataset(self, subject_files, test_trial_indices,
                        patch_idx, shared_cache):
        test_config = self.config.copy()
        test_config["patch_idx"] = patch_idx
        return WithinSubjectDatasetTotal(
            config=test_config,
            filepath=subject_files,
            trial_indices=test_trial_indices,
            shared_cache=shared_cache,
        )

    def predict_from_dataset(self, model, dataset, preloaded=None):
        """
        FP16 autocast + GPU 프리로드로 고속 추론.
        preloaded: (X_t, Y_t, mask_t) 튜플. 같은 dataset을 반복 호출 시 재사용.
        """
        model.eval()
        batch_size = self.config["batch_size"]

        if preloaded is not None:
            X_t, Y_t, mask_t = preloaded
        else:
            n = len(dataset)
            all_X, all_Y, all_mask = [], [], []
            for i in range(n):
                X, Y, stat, mask = dataset[i]
                all_X.append(X)
                all_Y.append(Y)
                all_mask.append(mask)
            X_t = torch.cat(all_X, dim=0)
            Y_t = torch.cat(all_Y, dim=0)
            mask_t = torch.cat(all_mask, dim=0)

        n = X_t.shape[0]
        all_preds, all_probs = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                x_b = X_t[s:e].cuda(0, non_blocking=True)
                m_b = mask_t[s:e].cuda(0, non_blocking=True)
                outputs = model(x_b, mask=m_b)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                probs = F.softmax(logits.float(), dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        return (torch.cat(all_preds).numpy(),
                Y_t.numpy(),
                torch.cat(all_probs).numpy())

    def _discover_trained_subjects(self):
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

    def _preload_dataset_to_tensors(self, dataset):
        """Dataset을 한 번에 텐서로 변환하여 캐싱 (반복 호출 방지)"""
        n = len(dataset)
        all_X, all_Y, all_mask = [], [], []
        for i in range(n):
            X, Y, stat, mask = dataset[i]
            all_X.append(X)
            all_Y.append(Y)
            all_mask.append(mask)
        return (torch.cat(all_X, dim=0),
                torch.cat(all_Y, dim=0),
                torch.cat(all_mask, dim=0))

    # ---------------------------------------------------
    # Single-subject inference (full TGM + diagonal CM)
    # ---------------------------------------------------
    def run_single_subject(self, subject_id, subject_files):
        """
        단일 피험자 full TGM inference (per-patch batched).
        - 모델이 mask[0,0]만 사용하므로 같은 patch의 샘플만 한 배치로 묶어야 함
        - 패치별 tensor를 미리 캐싱하여 I/O 최적화
        - CM은 diagonal (train_patch == test_patch) 에서만 생성
        Returns: list[dict] — subject_data rows
        """
        root_dir = self.config["ckpt_dir"]
        subject_dir = os.path.join(root_dir, subject_id)
        ckpt_paths = self._discover_subject_checkpoints(subject_dir)

        if not ckpt_paths:
            print(f"  [Skip] {subject_id}: checkpoint 없음")
            return []

        self.config["test_files"] = subject_files

        _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
            config=self.config, filepath=subject_files,
            val_size=0.2, test_size=0.2, random_state=42
        )

        input_len, input_ch = COMBDataset(
            config=self.config, filepath=subject_files
        )._get_sample_info()
        num_patches = self._calculate_num_patches(input_len)

        print(f"  Ckpts: {len(ckpt_paths)}, Test: {len(test_trial_indices)}, "
              f"Patches: {num_patches} (stride={self.stride})")
        print(f"  Mode: full TGM per-patch (CM = diagonal only)")

        # 모든 패치 tensor를 미리 캐싱
        shared_cache = {}
        tensor_cache = {}
        labels_np = None
        print(f"  [*] Preloading all {num_patches} test patch tensors...")
        for p_idx in tqdm(range(num_patches), desc="  Preload", unit="patch"):
            ds = self._create_dataset(
                subject_files, test_trial_indices, p_idx, shared_cache)
            X_t, Y_t, mask_t = self._preload_dataset_to_tensors(ds)
            tensor_cache[p_idx] = (X_t, Y_t, mask_t)
            if labels_np is None:
                labels_np = Y_t.numpy()
            del ds
        del shared_cache

        # 패치별 시간 정보 미리 계산
        patch_times = []
        for p_idx in range(num_patches):
            t_ms, _ = get_patch_time_ms_stride(
                p_idx, self.time_bin, self.stride,
                self.config['sampling_rate'])
            patch_times.append(t_ms)

        subject_data = []

        ckpt_pbar = tqdm(ckpt_paths, desc=f"  [{subject_id}] Ckpts", unit="ckpt")
        for ckpt_path in ckpt_pbar:
            ckpt_name = self._get_ckpt_name(ckpt_path)
            train_patch_idx = self._parse_patch_from_path(ckpt_path)
            epoch = self._parse_epoch_from_ckpt(ckpt_name)
            train_start_ms = patch_times[train_patch_idx]

            ckpt_pbar.set_description(f"  P{train_patch_idx} ep{epoch}")

            save_dir = os.path.dirname(ckpt_path).replace("checkpoints", "analysis")

            if not self.config.get("skip_pred", False):
                model = self._load_model_instance(ckpt_path, self.model)
                if model is None:
                    continue
            else:
                model = self.model

            # 패치별 개별 inference (mask[0,0] 이슈 방지)
            for test_patch_idx in range(num_patches):
                preloaded = tensor_cache[test_patch_idx]

                preds, _, probs = self.predict_from_dataset(
                    model, None, preloaded=preloaded)

                metrics = self.evaluator.compute_metrics(
                    preds, labels_np, probs,
                    params=self.config["metrics"])

                # CM은 diagonal에서만 생성
                if test_patch_idx == train_patch_idx:
                    self.visualizer.plot_cm(
                        save_dir, labels_np, preds,
                        train_patch_idx, test_patch_idx,
                        metrics['acc'], ckpt_name=ckpt_name)

                subject_data.append({
                    'subject': subject_id,
                    'ckpt_name': ckpt_name,
                    'epoch': epoch,
                    'train_patch_idx': train_patch_idx,
                    'train_time_ms': train_start_ms,
                    'test_patch_idx': test_patch_idx,
                    'test_time_ms': patch_times[test_patch_idx],
                    'test_acc': metrics['acc'],
                    'test_bal_acc': metrics['bal_acc'],
                })

        del tensor_cache

        # Per-subject summary CSV + TGM + temporal accuracy plot
        if subject_data:
            df_subj = pd.DataFrame(subject_data)
            subj_analysis_dir = os.path.join(root_dir, subject_id, "combined_analysis")
            os.makedirs(subj_analysis_dir, exist_ok=True)

            self.recorder.save_summary_csv(
                df_subj, save_dir=subj_analysis_dir,
                ckpt_name=f"{subject_id}_all")

            # Per-subject TGM (epoch 평균)
            df_tgm_avg = df_subj.groupby(
                ['train_patch_idx', 'test_patch_idx']
            ).agg({
                'train_time_ms': 'first',
                'test_time_ms': 'first',
                'test_acc': 'mean',
                'test_bal_acc': 'mean',
            }).reset_index()

            self.visualizer.plot_tgm(
                df_tgm_avg, metric_key='test_acc',
                save_dir=subj_analysis_dir,
                title=f"{subject_id} - TGM (test_acc, epoch avg)",
                filename_prefix=f"{subject_id}_tgm")

            self.visualizer.plot_tgm(
                df_tgm_avg, metric_key='test_bal_acc',
                save_dir=subj_analysis_dir,
                title=f"{subject_id} - TGM (test_bal_acc, epoch avg)",
                filename_prefix=f"{subject_id}_tgm")

            # Diagonal temporal accuracy
            df_diag = df_subj[df_subj['train_patch_idx'] == df_subj['test_patch_idx']]
            self.visualizer.plot_temporal_accuracy(
                df_diag, save_dir=subj_analysis_dir,
                title=f"{subject_id} - Diagonal Accuracy over Time",
                filename_prefix=f"{subject_id}_diagonal",
                n_classes=self.config.get("n_classes", 6))

            print(f"  [Done] {subject_id} inference 완료 → {subj_analysis_dir}")

        return subject_data

    # ---------------------------------------------------
    # Cross-subject TGM from summary CSVs on disk
    # ---------------------------------------------------
    def generate_cross_subject_tgm(self):
        """
        각 피험자의 combined_analysis/{subject}_all_summary.csv를
        디스크에서 읽어 합친 뒤 cross-subject TGM 생성.
        inference 결과가 이미 있으면 언제든 실행 가능.
        """
        root_dir = self.config["ckpt_dir"]
        dfs = []

        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path) or not entry.startswith("sub-"):
                continue
            csv_path = os.path.join(entry_path, "combined_analysis",
                                    f"{entry}_all_summary.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                dfs.append(df)
                print(f"  [Loaded] {csv_path} ({len(df)} rows)")

        if not dfs:
            print("[!] No summary CSVs found.")
            return

        df_all = pd.concat(dfs, ignore_index=True)
        n_total_subjects = df_all['subject'].nunique()
        print(f"\n[*] Loaded {n_total_subjects} subjects, {len(df_all)} total rows")

        combined_dir = os.path.join(root_dir, "cross_subject_analysis")
        os.makedirs(combined_dir, exist_ok=True)

        # Save merged summary
        self.recorder.save_summary_csv(
            df_all, save_dir=combined_dir,
            ckpt_name="all_subjects_all_epochs")

        # Per-epoch cross-subject averaged TGM
        epochs = sorted(df_all['epoch'].unique())
        print(f"[*] Cross-subject epoch-averaged TGMs for {len(epochs)} epochs...")

        for epoch_num in epochs:
            df_epoch = df_all[df_all['epoch'] == epoch_num]
            epoch_label = f"epoch_{epoch_num:02d}"

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
                df_avg, metric_key='test_acc',
                save_dir=epoch_dir,
                title=(f"Epoch {epoch_num} - Cross-Subject Avg TGM "
                       f"(test_acc, n={n_subjects})"),
                filename_prefix=f"{epoch_label}_cross_subject_avg")

            self.visualizer.plot_tgm(
                df_avg, metric_key='test_bal_acc',
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

        self.visualizer.plot_tgm(
            df_overall_avg, metric_key='test_acc',
            save_dir=combined_dir,
            title=(f"All Epochs All Subjects Avg TGM "
                   f"(test_acc, n={n_total_subjects})"),
            filename_prefix="all_epochs_all_subjects_avg")

        self.visualizer.plot_tgm(
            df_overall_avg, metric_key='test_bal_acc',
            save_dir=combined_dir,
            title=(f"All Epochs All Subjects Avg TGM "
                   f"(test_bal_acc, n={n_total_subjects})"),
            filename_prefix="all_epochs_all_subjects_avg")

        print(f"\n[!] Cross-Subject Analysis Complete. Output: {combined_dir}")

    # ---------------------------------------------------
    # Full run: per-subject inference → cross-subject TGM
    # ---------------------------------------------------
    def run(self, target_subjects=None):
        """
        전체 또는 지정 피험자 inference + cross-subject TGM.
        Args:
            target_subjects: list[str] or None.
                None이면 학습된 전체 피험자, 아니면 지정 피험자만.
        """
        subject_files_map = discover_all_subjects(self.config["data_dir"])
        trained_subjects = self._discover_trained_subjects()

        if target_subjects:
            subject_ids = [s for s in target_subjects if s in trained_subjects]
            skipped = [s for s in target_subjects if s not in trained_subjects]
            if skipped:
                print(f"[Warning] checkpoint 없는 피험자 건너뜀: {skipped}")
        else:
            subject_ids = sorted(trained_subjects.keys())

        print(f"[*] Inference 대상: {len(subject_ids)} subjects")

        # Resume: skip subjects that already have summary CSVs
        root_dir = self.config["ckpt_dir"]
        done_subjects = set()
        for sid in subject_ids:
            csv_path = os.path.join(root_dir, sid, "combined_analysis",
                                    f"{sid}_all_summary.csv")
            if os.path.exists(csv_path):
                done_subjects.add(sid)
        if done_subjects:
            print(f"[*] Resume: {len(done_subjects)} subjects already done, skipping")

        subj_pbar = tqdm(subject_ids, desc="Subjects", unit="subj")
        for subject_id in subj_pbar:
            subj_pbar.set_description(f"Subject {subject_id}")

            if subject_id in done_subjects:
                continue

            if subject_id not in subject_files_map:
                print(f"  [Skip] No data files found for {subject_id}")
                continue

            subj_files = subject_files_map[subject_id]
            self.run_single_subject(subject_id, subj_files)
            torch.cuda.empty_cache()

        # Cross-subject TGM from saved summary CSVs
        self.generate_cross_subject_tgm()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="EEGPT Within-Subject Inference (Total)")
    parser.add_argument(
        '--subject', nargs='+', default=None,
        help='특정 피험자만 inference (e.g. --subject sub-14 sub-15)')
    parser.add_argument(
        '--cross-subject-only', action='store_true',
        help='inference 건너뛰고 기존 summary CSV로 cross-subject TGM만 생성')
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='inference batch size (default: config 값 사용)')
    args = parser.parse_args()

    if is_debugging():
        print("[Debug Mode]")
        num_workers = 1
    else:
        print("[Inference Mode]")
        num_workers = 16

    STRIDE = 4
    TIME_BIN = 16

    test_config = {
        "data_dir": "./EEG(500Hz)_53ch/",
        "batch_size": 2048,
        "num_workers": num_workers,
        "shuffle": False,
        "sampling_rate": 500,
        "start_time_ms": -200,
        "data_ext": "npy",
        "window_type": "fixed",
        "time_bin": TIME_BIN,
        "file_chunk_type": "subject",
        "normalize_method": "zscore",
        "patch_idx": None,
        "stride": STRIDE,
        "save_dir": f"EEGPT/within_total/500Hz_t{TIME_BIN}_s{STRIDE}",
        "num_epochs": 100,
        "patience": 10,
        "n_classes": 6,
        "is_label_null": True,
        "skip_pred": False,
        "metrics": ["acc", "bal_acc"],
        "ckpt_dir": f"./EEGPT/within_total/500Hz_t{TIME_BIN}_s{STRIDE}",
        "csv_input_dir": None,
        "test_files": None,
    }

    # Sample file for model init
    subject_files_map = discover_all_subjects(test_config["data_dir"])
    first_subject = sorted(subject_files_map.keys())[0]
    first_files = subject_files_map[first_subject]
    test_config["test_files"] = first_files

    input_len, input_ch = COMBDataset(
        config=test_config, filepath=first_files
    )._get_sample_info()
    n_patches = (input_len - TIME_BIN) // STRIDE + 1

    test_config["n_patches"] = n_patches
    test_config["input_len"] = input_len
    test_config["input_ch"] = input_ch

    model = LitEEGPTCausal_LoRA(
        config=test_config,
        fixed_train_patch_idx=0,
    )

    if args.batch_size:
        test_config["batch_size"] = args.batch_size

    manager = EEGPTWithinInferenceTotal(test_config, model=model)

    if args.cross_subject_only:
        print("[*] Cross-subject TGM only (from existing summary CSVs)")
        manager.generate_cross_subject_tgm()
    else:
        manager.run(target_subjects=args.subject)

    del model, manager
    torch.cuda.empty_cache()


