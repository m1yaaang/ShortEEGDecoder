"""
EEGNet Within-Subject Inference (Total)
- Per-subject full TGM (all train_patch × test_patch)
- CM은 diagonal (train_patch == test_patch) 만 생성
- FP16 autocast + tensor 캐싱으로 고속 inference
- Per-patch batching (모델이 mask[0,0] 사용하므로 mega-batch 불가)

Usage:
  python EEGNet/EEGNet_inference_total.py
  python EEGNet/EEGNet_inference_total.py --subject sub-22 sub-23
  python EEGNet/EEGNet_inference_total.py --cross-subject-only
  python EEGNet/EEGNet_inference_total.py --batch_size 4096
"""
import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

from utils_data import (discover_all_subjects, WithinSubjectDatasetTotal,
                        get_patch_time_ms_stride, COMBDataset)
from utils_infer import Evaluator, Recorder, Visualizer
from EEGNet.EEGNet_total import EEGNet


def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None


class EEGNetWithinInferenceTotal:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.time_bin = config["time_bin"]
        self.stride = config.get("stride") or config["time_bin"]

        self.evaluator = Evaluator(config)
        self.recorder = Recorder(config)
        self.visualizer = Visualizer(config)

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
        return int(match.group(1)) if match else 0

    def _parse_epoch_from_ckpt(self, ckpt_name):
        match = re.search(r'epoch=(\d+)', ckpt_name)
        return int(match.group(1)) if match else -1

    def _get_ckpt_name(self, ckpt_path):
        return os.path.basename(ckpt_path).replace(".ckpt", "")

    def _load_model(self, ckpt_path):
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

        # Strip 'model.' prefix if needed (Lightning wrapper)
        sample_key = next(iter(state.keys()), '')
        net_keys = set(self.model.state_dict().keys())
        if sample_key.startswith('model.') and not any(
                k.startswith('model.') for k in net_keys):
            state = {k[6:]: v for k, v in state.items()
                     if k.startswith('model.')}

        self.model.load_state_dict(state, strict=False)
        self.model.cuda().eval()
        return self.model

    def _create_dataset(self, subject_files, trial_indices, patch_idx,
                        shared_cache):
        cfg = self.config.copy()
        cfg["patch_idx"] = patch_idx
        return WithinSubjectDatasetTotal(
            config=cfg, filepath=subject_files,
            trial_indices=trial_indices, shared_cache=shared_cache)

    def _preload_to_tensors(self, dataset):
        all_X, all_Y, all_mask = [], [], []
        for i in range(len(dataset)):
            X, Y, stat, mask = dataset[i]
            all_X.append(X)
            all_Y.append(Y)
            all_mask.append(mask)
        return (torch.cat(all_X, dim=0),
                torch.cat(all_Y, dim=0),
                torch.cat(all_mask, dim=0))

    def _predict(self, model, preloaded):
        """FP16 autocast + preloaded tensor → fast inference."""
        model.eval()
        X_t, Y_t, mask_t = preloaded
        n = X_t.shape[0]
        batch_size = self.config["batch_size"]
        patch_size = self.config.get("patch_size")

        all_preds, all_probs = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                x_b = X_t[s:e].cuda(0, non_blocking=True)
                m_b = mask_t[s:e].cuda(0, non_blocking=True)
                outputs = model(x_b, mask=m_b, patch_size=patch_size)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                probs = F.softmax(logits.float(), dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        return (torch.cat(all_preds).numpy(),
                Y_t.numpy(),
                torch.cat(all_probs).numpy())

    # ----------------------------------------------------------
    # Per-subject inference: full TGM + diagonal CM
    # ----------------------------------------------------------
    def run_single_subject(self, subject_id, subject_files):
        root_dir = self.config["ckpt_dir"]
        subject_dir = os.path.join(root_dir, subject_id)
        ckpt_paths = self._discover_subject_checkpoints(subject_dir)

        if not ckpt_paths:
            print(f"  [Skip] {subject_id}: no checkpoints")
            return []

        self.config["test_files"] = subject_files

        _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
            config=self.config, filepath=subject_files,
            val_size=0.2, test_size=0.2, random_state=42)

        input_len, input_ch = COMBDataset(
            config=self.config, filepath=subject_files
        )._get_sample_info()
        num_patches = self._calculate_num_patches(input_len)

        print(f"  Ckpts: {len(ckpt_paths)}, Test: {len(test_trial_indices)}, "
              f"Patches: {num_patches} (stride={self.stride})")
        print(f"  Mode: full TGM per-patch (CM = diagonal only)")

        # Preload all test patch tensors
        shared_cache = {}
        tensor_cache = {}
        labels_np = None
        print(f"  [*] Preloading {num_patches} test patches...")
        for p_idx in tqdm(range(num_patches), desc="  Preload", unit="patch"):
            ds = self._create_dataset(
                subject_files, test_trial_indices, p_idx, shared_cache)
            X_t, Y_t, mask_t = self._preload_to_tensors(ds)
            tensor_cache[p_idx] = (X_t, Y_t, mask_t)
            if labels_np is None:
                labels_np = Y_t.numpy()
            del ds
        del shared_cache

        # Patch time info
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

            model = self._load_model(ckpt_path)
            if model is None:
                continue

            for test_patch_idx in range(num_patches):
                preloaded = tensor_cache[test_patch_idx]
                preds, _, probs = self._predict(model, preloaded)

                metrics = self.evaluator.compute_metrics(
                    preds, labels_np, probs,
                    params=self.config["metrics"])

                # CM diagonal only
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

        # Per-subject outputs
        if subject_data:
            df_subj = pd.DataFrame(subject_data)
            subj_dir = os.path.join(root_dir, subject_id, "combined_analysis")
            os.makedirs(subj_dir, exist_ok=True)

            self.recorder.save_summary_csv(
                df_subj, save_dir=subj_dir,
                ckpt_name=f"{subject_id}_all")

            # Best epoch selection: per train_patch, highest diagonal acc
            df_best = self._extract_best_epoch(df_subj)
            self.recorder.save_summary_csv(
                df_best, save_dir=subj_dir,
                ckpt_name=f"{subject_id}_best_epoch")

            # Best-epoch TGM
            for m in ['test_acc', 'test_bal_acc']:
                self.visualizer.plot_tgm(
                    ckpt=f"{subject_id}_best",
                    df=df_best, metric_key=m,
                    save_dir=subj_dir,
                    title=f"{subject_id} - TGM ({m}, best epoch)",
                    filename_prefix=f"{subject_id}_best_epoch_tgm")

            # Best-epoch diagonal plot (plot_best_epoch_diagonal style)
            self._plot_diagonal_best_epoch(
                df_best, subject_id, subj_dir)

            print(f"  [Done] {subject_id} → {subj_dir}")

        return subject_data

    @staticmethod
    def _parse_valid_loss(ckpt_name):
        """ckpt_name에서 valid_loss 파싱. 없으면 inf 반환."""
        match = re.search(r'valid_loss[=_]([\d.]+)', ckpt_name)
        return float(match.group(1)) if match else float('inf')

    def _extract_best_epoch(self, df_subj):
        """각 train_patch별 valid_loss가 가장 낮은 checkpoint 선택 후 전체 TGM row 반환."""
        # ckpt_name별 valid_loss 파싱
        ckpt_names = df_subj['ckpt_name'].unique()
        vloss_map = {c: self._parse_valid_loss(c) for c in ckpt_names}
        df_subj = df_subj.copy()
        df_subj['_valid_loss'] = df_subj['ckpt_name'].map(vloss_map)

        # 각 train_patch별 valid_loss 최소인 checkpoint 선택
        diag = df_subj[df_subj['train_patch_idx'] == df_subj['test_patch_idx']]
        best_idx = diag.groupby('train_patch_idx')['_valid_loss'].idxmin()
        best_ckpts = df_subj.loc[best_idx, ['train_patch_idx', 'ckpt_name']].values.tolist()
        best_map = {int(p): c for p, c in best_ckpts}

        rows = []
        for train_p, ckpt in best_map.items():
            mask = ((df_subj['train_patch_idx'] == train_p) &
                    (df_subj['ckpt_name'] == ckpt))
            rows.append(df_subj[mask])
        df_best = pd.concat(rows, ignore_index=True)
        df_best.drop(columns=['_valid_loss'], inplace=True)
        return df_best

    def _plot_diagonal_best_epoch(self, df_best, subject_id, save_dir,
                                   filename_prefix=None):
        """Best-epoch diagonal accuracy plot (plot_best_epoch_diagonal.py style)."""
        diag = df_best[df_best['train_patch_idx'] == df_best['test_patch_idx']].copy()
        diag = diag.sort_values('train_patch_idx').reset_index(drop=True)
        time_ms = diag['train_time_ms'].values
        chance = 100.0 / self.config.get("n_classes", 6)
        prefix = filename_prefix or subject_id

        for m in ['test_acc', 'test_bal_acc']:
            values = diag[m].values
            best_idx = np.argmax(values)
            best_time = time_ms[best_idx]
            best_val = values[best_idx]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(time_ms, values, '-', color='steelblue', linewidth=1.5,
                    label='best-epoch diag')
            ax.axhline(chance, color='gray', linestyle='--', linewidth=1.2,
                       alpha=0.7, label=f'chance={chance:.1f}%')
            ax.axvline(best_time, color='gray', linestyle='--', linewidth=1.0,
                       alpha=0.5)

            y_min, y_max = ax.get_ylim()
            if best_val > (y_min + y_max) / 2:
                xytext, va = (10, -35), 'top'
            else:
                xytext, va = (10, 15), 'bottom'

            ax.plot(best_time, best_val, 'rv', markersize=10, zorder=5)
            ax.annotate(
                f'P{best_idx} ({best_time:.0f}ms, {best_val:.1f}%)',
                xy=(best_time, best_val), xytext=xytext,
                textcoords='offset points', fontsize=9,
                color='red', fontweight='bold', va=va,
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(m)
            ax.set_title(f"{subject_id} - Best-Epoch Diagonal {m}")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir,
                        f"{prefix}_best_epoch_diagonal_{m}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ----------------------------------------------------------
    # Cross-subject TGM from saved CSVs
    # ----------------------------------------------------------
    def generate_cross_subject_tgm(self):
        root_dir = self.config["ckpt_dir"]
        dfs = []
        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path) or not entry.startswith("sub-"):
                continue
            csv_path = os.path.join(entry_path, "combined_analysis",
                                    f"{entry}_best_epoch_summary.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                dfs.append(df)
                print(f"  [Loaded] {csv_path} ({len(df)} rows)")

        if not dfs:
            print("[!] No best_epoch_summary CSVs found.")
            return

        df_all = pd.concat(dfs, ignore_index=True)
        n_subj = df_all['subject'].nunique()
        print(f"\n[*] {n_subj} subjects, {len(df_all)} total rows (best epoch)")

        combined_dir = os.path.join(root_dir, "cross_subject_analysis")
        os.makedirs(combined_dir, exist_ok=True)

        self.recorder.save_summary_csv(
            df_all, save_dir=combined_dir,
            ckpt_name="all_subjects_best_epoch")

        # Cross-subject average TGM (best epoch only)
        df_avg = df_all.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first', 'test_time_ms': 'first',
            'test_acc': 'mean', 'test_bal_acc': 'mean',
        }).reset_index()

        last_ckpt = df_all['ckpt_name'].iloc[-1]
        for m in ['test_acc', 'test_bal_acc']:
            self.visualizer.plot_tgm(
                ckpt=last_ckpt, df=df_avg, metric_key=m,
                save_dir=combined_dir,
                title=f"Cross-Subject Best-Epoch Avg ({m}, n={n_subj})",
                filename_prefix="cross_subject_best_epoch_avg")

        # Cross-subject diagonal plot
        self._plot_diagonal_best_epoch(
            df_avg, f"All Subjects (n={n_subj})", combined_dir,
            filename_prefix="cross_subject")

        print(f"\n[!] Cross-Subject Analysis → {combined_dir}")

    # ----------------------------------------------------------
    # Cross-subject CM: best-epoch 모델로 re-inference → 전체 피험자 CM
    # ----------------------------------------------------------
    def generate_cross_subject_cm(self):
        """Cross-subject confusion matrix: per-patch best-epoch predictions aggregated."""
        root_dir = self.config["ckpt_dir"]
        output_dir = os.path.join(root_dir, "avg_cm")
        os.makedirs(output_dir, exist_ok=True)

        subject_files_map = discover_all_subjects(self.config["data_dir"])
        n_classes = self.config.get("n_classes", 6)

        # Get patch info
        first_subject = sorted(subject_files_map.keys())[0]
        first_files = subject_files_map[first_subject]
        self.config["test_files"] = first_files
        input_len, _ = COMBDataset(
            config=self.config, filepath=first_files)._get_sample_info()
        num_patches = self._calculate_num_patches(input_len)

        patch_times = {}
        for p_idx in range(num_patches):
            t_ms, _ = get_patch_time_ms_stride(
                p_idx, self.time_bin, self.stride,
                self.config['sampling_rate'])
            patch_times[p_idx] = t_ms

        # Discover subjects
        subjects = sorted([d for d in os.listdir(root_dir)
                           if d.startswith("sub-") and
                           os.path.isdir(os.path.join(root_dir, d))])

        print(f"\n[*] Cross-Subject CM: {len(subjects)} subjects, {num_patches} patches")

        patch_preds = defaultdict(list)
        patch_labels = defaultdict(list)

        for subj in tqdm(subjects, desc="CM Subjects"):
            csv_path = os.path.join(root_dir, subj, "combined_analysis",
                                    f"{subj}_all_summary.csv")
            if not os.path.exists(csv_path):
                continue
            if subj not in subject_files_map:
                continue

            df = pd.read_csv(csv_path)
            diag = df[df['train_patch_idx'] == df['test_patch_idx']].copy()
            diag['_valid_loss'] = diag['ckpt_name'].apply(self._parse_valid_loss)
            best = diag.loc[diag.groupby('train_patch_idx')['_valid_loss'].idxmin()]
            best_entries = best[['train_patch_idx', 'epoch', 'ckpt_name']].values.tolist()

            subject_files = subject_files_map[subj]
            self.config["test_files"] = subject_files

            _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
                config=self.config, filepath=subject_files,
                val_size=0.2, test_size=0.2, random_state=42)

            subject_dir = os.path.join(root_dir, subj)
            shared_cache = {}

            for train_patch_idx, epoch, ckpt_name in best_entries:
                train_patch_idx = int(train_patch_idx)
                # Find checkpoint
                ckpt_path = None
                for exp_name in os.listdir(subject_dir):
                    ckpt_dir_path = os.path.join(subject_dir, exp_name, "checkpoints")
                    if not os.path.exists(ckpt_dir_path):
                        continue
                    full_path = os.path.join(ckpt_dir_path, ckpt_name + ".ckpt")
                    if os.path.exists(full_path):
                        ckpt_path = full_path
                        break
                if ckpt_path is None:
                    continue

                model = self._load_model(ckpt_path)
                if model is None:
                    continue

                ds = self._create_dataset(
                    subject_files, test_trial_indices,
                    train_patch_idx, shared_cache)
                preloaded = self._preload_to_tensors(ds)
                del ds

                preds, labels, _ = self._predict(model, preloaded)
                patch_preds[train_patch_idx].append(preds)
                patch_labels[train_patch_idx].append(labels)
                del preloaded

            del shared_cache
            torch.cuda.empty_cache()

        # Generate per-patch CMs
        print(f"\n[*] Generating per-patch CMs...")
        label_names = np.arange(n_classes)
        all_preds_global = []
        all_labels_global = []
        patch_results = []

        for p_idx in sorted(patch_preds.keys()):
            preds_all = np.concatenate(patch_preds[p_idx])
            labels_all = np.concatenate(patch_labels[p_idx])
            n_subj = len(patch_preds[p_idx])

            cm = confusion_matrix(labels_all, preds_all, labels=label_names)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm.astype(float) / row_sums * 100

            acc = accuracy_score(labels_all, preds_all) * 100
            bal_acc = balanced_accuracy_score(labels_all, preds_all) * 100
            time_ms = patch_times.get(p_idx, 0)

            fig, ax = plt.subplots(figsize=(8, 7))
            cm_vmin = np.floor(cm_norm.min() / 5) * 5
            cm_vmax = np.ceil(cm_norm.max() / 5) * 5
            sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                        vmin=cm_vmin, vmax=cm_vmax,
                        xticklabels=[f"Class_{i}" for i in label_names],
                        yticklabels=[f"Class_{i}" for i in label_names])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f"P{p_idx} ({time_ms:.0f}ms) Best-Epoch CM (n={n_subj})\n"
                         f"Acc={acc:.2f}%, Bal_Acc={bal_acc:.2f}%")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f"P{p_idx:03d}_{time_ms:.0f}ms_cm_norm.png"), dpi=150)
            plt.close()

            patch_results.append({
                'patch_idx': p_idx, 'time_ms': time_ms,
                'acc': acc, 'bal_acc': bal_acc, 'n_subjects': n_subj
            })
            all_preds_global.extend(preds_all)
            all_labels_global.extend(labels_all)

        # Overall CM
        all_preds_global = np.array(all_preds_global)
        all_labels_global = np.array(all_labels_global)
        cm_total = confusion_matrix(all_labels_global, all_preds_global,
                                    labels=label_names)
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_total_norm = cm_total.astype(float) / row_sums * 100

        acc_total = accuracy_score(all_labels_global, all_preds_global) * 100
        bal_acc_total = balanced_accuracy_score(
            all_labels_global, all_preds_global) * 100

        fig, ax = plt.subplots(figsize=(8, 7))
        cm_vmin = np.floor(cm_total_norm.min() / 5) * 5
        cm_vmax = np.ceil(cm_total_norm.max() / 5) * 5
        sns.heatmap(cm_total_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                    vmin=cm_vmin, vmax=cm_vmax,
                    xticklabels=[f"Class_{i}" for i in label_names],
                    yticklabels=[f"Class_{i}" for i in label_names])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"All Patches Best-Epoch CM (n={len(subjects)})\n"
                     f"Acc={acc_total:.2f}%, Bal_Acc={bal_acc_total:.2f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    "all_patches_overall_cm_norm.png"), dpi=150)
        plt.close()

        # Summary CSV
        df_results = pd.DataFrame(patch_results)
        df_results.to_csv(os.path.join(output_dir,
                          "per_patch_cm_summary.csv"), index=False)

        print(f"\n[*] Overall: acc={acc_total:.2f}%, bal_acc={bal_acc_total:.2f}%")
        print(f"[*] Generated {len(patch_results)} per-patch CMs + 1 overall CM")
        print(f"[!] CM Output → {output_dir}")

    # ----------------------------------------------------------
    # Reprocess: 기존 CSV에서 best-epoch 분석만 재생성
    # ----------------------------------------------------------
    def reprocess_subject(self, subject_id):
        root_dir = self.config["ckpt_dir"]
        subj_dir = os.path.join(root_dir, subject_id, "combined_analysis")
        csv_path = os.path.join(subj_dir, f"{subject_id}_all_summary.csv")
        df_subj = pd.read_csv(csv_path)
        print(f"  [Reprocess] {subject_id}: {len(df_subj)} rows from CSV")

        df_best = self._extract_best_epoch(df_subj)
        self.recorder.save_summary_csv(
            df_best, save_dir=subj_dir,
            ckpt_name=f"{subject_id}_best_epoch")

        for m in ['test_acc', 'test_bal_acc']:
            self.visualizer.plot_tgm(
                ckpt=f"{subject_id}_best",
                df=df_best, metric_key=m,
                save_dir=subj_dir,
                title=f"{subject_id} - TGM ({m}, best epoch)",
                filename_prefix=f"{subject_id}_best_epoch_tgm")

        self._plot_diagonal_best_epoch(df_best, subject_id, subj_dir)
        print(f"  [Done] {subject_id} → {subj_dir}")

    # ----------------------------------------------------------
    # Run all subjects
    # ----------------------------------------------------------
    def run(self, target_subjects=None, force=False):
        subject_files_map = discover_all_subjects(self.config["data_dir"])
        root_dir = self.config["ckpt_dir"]

        # Find trained subjects
        trained = {}
        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path) or not entry.startswith("sub-"):
                continue
            ckpts = self._discover_subject_checkpoints(entry_path)
            if ckpts:
                trained[entry] = True

        subject_ids = sorted(trained.keys())
        if target_subjects:
            subject_ids = [s for s in subject_ids if s in target_subjects]

        print(f"[*] Inference 대상: {len(subject_ids)} subjects")

        subj_pbar = tqdm(subject_ids, desc="Subjects", unit="subj")
        for subject_id in subj_pbar:
            subj_pbar.set_description(f"Subject {subject_id}")

            if subject_id not in subject_files_map:
                print(f"  [Skip] {subject_id}: no data files")
                continue

            subj_dir = os.path.join(root_dir, subject_id, "combined_analysis")
            all_csv = os.path.join(subj_dir, f"{subject_id}_all_summary.csv")
            best_csv = os.path.join(subj_dir, f"{subject_id}_best_epoch_summary.csv")

            if os.path.exists(all_csv) and not force:
                if os.path.exists(best_csv):
                    print(f"  [Skip] {subject_id}: already done")
                    continue
                # CSV exists but no best-epoch analysis → reprocess only
                self.reprocess_subject(subject_id)
            else:
                self.run_single_subject(subject_id, subject_files_map[subject_id])
                torch.cuda.empty_cache()

        self.generate_cross_subject_tgm()
        self.generate_cross_subject_cm()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="EEGNet Within-Subject Inference (Total)")
    parser.add_argument(
        '--subject', nargs='+', default=None,
        help='특정 피험자만 (e.g. --subject sub-22 sub-23)')
    parser.add_argument(
        '--cross-subject-only', action='store_true',
        help='기존 CSV로 cross-subject TGM만 생성')
    parser.add_argument(
        '--force', action='store_true',
        help='기존 CSV 무시하고 전체 재실행')
    parser.add_argument(
        '--batch_size', type=int, default=4096,
        help='inference batch size (default: 4096)')
    parser.add_argument(
        '--patch_size', type=int, default=64,
        help='patch interpolation size (default: 64)')
    parser.add_argument(
        '--stride', type=int, default=4)
    parser.add_argument(
        '--time_bin', type=int, default=16)
    parser.add_argument(
        '--num_workers', type=int, default=16)
    parser.add_argument(
        '--data_dir', default='./EEG(500Hz)_53ch/')
    args = parser.parse_args()

    if is_debugging():
        print("[Debug Mode]")
        num_workers = 1
    else:
        print("[Inference Mode]")
        num_workers = args.num_workers

    STRIDE = args.stride
    TIME_BIN = args.time_bin
    PATCH_SIZE = args.patch_size

    log_dir = f"./EEGNet/within_logs/500Hz_t{TIME_BIN}_s{STRIDE}_w{PATCH_SIZE}"

    test_config = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
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
        "save_dir": log_dir,
        "num_epochs": 500,
        "patience": 10,
        "n_classes": 6,
        "is_label_null": True,
        "skip_pred": False,
        "metrics": ["acc", "bal_acc"],
        "ckpt_dir": log_dir,
        "csv_input_dir": None,
        "test_files": None,
        "patch_size": PATCH_SIZE,
    }

    # Model init (architecture only — weights loaded per checkpoint)
    subject_files_map = discover_all_subjects(test_config["data_dir"])
    first_subject = sorted(subject_files_map.keys())[0]
    first_files = subject_files_map[first_subject]
    test_config["test_files"] = first_files

    input_len, input_ch = COMBDataset(
        config=test_config, filepath=first_files
    )._get_sample_info()

    test_config["input_len"] = input_len
    test_config["input_ch"] = input_ch

    model = EEGNet(
        n_channels=input_ch, n_timepoints=PATCH_SIZE,
        n_classes=test_config["n_classes"]).cuda(0)

    manager = EEGNetWithinInferenceTotal(test_config, model=model)

    if args.cross_subject_only:
        manager.generate_cross_subject_tgm()
        manager.generate_cross_subject_cm()
    else:
        manager.run(target_subjects=args.subject, force=args.force)

    del model, manager
    torch.cuda.empty_cache()
