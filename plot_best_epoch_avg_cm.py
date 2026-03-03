"""
전체 subject의 best-epoch diagonal confusion matrix를 patch별로 생성.
각 patch(train==test)마다 전체 subject의 predictions를 모아 CM을 생성.

Usage:
  python plot_best_epoch_avg_cm.py --root EEGNet/within_logs/500Hz_t16_s4_w64
  python plot_best_epoch_avg_cm.py --root EEGNet/within_logs/500Hz_t16_s4_w64 --subject sub-22 sub-70
  python plot_best_epoch_avg_cm.py --root EEGNet/within_logs/500Hz_t16_s4_w64 --output_dir EEGNet/within_logs/500Hz_t16_s4_w64/avg_cm
"""
import argparse
import os
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
from EEGNet.EEGNet_total import EEGNet


def _parse_valid_loss(ckpt_name):
    """ckpt_name에서 valid_loss 파싱."""
    match = re.search(r'valid_loss[=_]([\d.]+)', ckpt_name)
    return float(match.group(1)) if match else float('inf')


def extract_best_epoch_diag(df):
    """각 train_patch별 valid_loss가 가장 낮은 checkpoint의 (train_patch_idx, epoch, ckpt_name) 반환."""
    diag = df[df['train_patch_idx'] == df['test_patch_idx']].copy()
    diag['_valid_loss'] = diag['ckpt_name'].apply(_parse_valid_loss)
    best = diag.loc[diag.groupby('train_patch_idx')['_valid_loss'].idxmin()]
    return best[['train_patch_idx', 'epoch', 'ckpt_name']].values.tolist()


def find_ckpt_path(subject_dir, ckpt_name):
    """subject_dir 내에서 ckpt_name에 해당하는 .ckpt 파일 경로를 찾는다."""
    for exp_name in os.listdir(subject_dir):
        ckpt_dir = os.path.join(subject_dir, exp_name, "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue
        ckpt_file = ckpt_name + ".ckpt"
        full_path = os.path.join(ckpt_dir, ckpt_file)
        if os.path.exists(full_path):
            return full_path
    return None


def load_model(model, ckpt_path):
    """체크포인트에서 가중치를 로드."""
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

    sample_key = next(iter(state.keys()), '')
    net_keys = set(model.state_dict().keys())
    if sample_key.startswith('model.') and not any(
            k.startswith('model.') for k in net_keys):
        state = {k[6:]: v for k, v in state.items() if k.startswith('model.')}

    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


def preload_patch_tensors(config, subject_files, trial_indices, patch_idx, shared_cache):
    """단일 patch의 test data를 텐서로 프리로드."""
    cfg = config.copy()
    cfg["patch_idx"] = patch_idx
    ds = WithinSubjectDatasetTotal(
        config=cfg, filepath=subject_files,
        trial_indices=trial_indices, shared_cache=shared_cache)
    all_X, all_Y, all_mask = [], [], []
    for i in range(len(ds)):
        X, Y, stat, mask = ds[i]
        all_X.append(X)
        all_Y.append(Y)
        all_mask.append(mask)
    del ds
    return (torch.cat(all_X, dim=0),
            torch.cat(all_Y, dim=0),
            torch.cat(all_mask, dim=0))


def predict(model, preloaded, batch_size, patch_size):
    """FP16 autocast inference."""
    X_t, Y_t, mask_t = preloaded
    n = X_t.shape[0]
    all_preds = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            x_b = X_t[s:e].cuda(0, non_blocking=True)
            m_b = mask_t[s:e].cuda(0, non_blocking=True)
            outputs = model(x_b, mask=m_b, patch_size=patch_size)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds).numpy(), Y_t.numpy()


def plot_cm(cm, cm_norm, acc, bal_acc, patch_idx, time_ms, n_subj, n_classes,
            save_dir):
    """단일 patch에 대한 CM 저장 (count + normalized)."""
    label_names = np.arange(n_classes)

    # Normalized CM
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=[f"Class_{i}" for i in label_names],
                yticklabels=[f"Class_{i}" for i in label_names])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"P{patch_idx} ({time_ms:.0f}ms) Best-Epoch CM (n={n_subj})\n"
                 f"Acc={acc:.2f}%, Bal_Acc={bal_acc:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                f"P{patch_idx:03d}_{time_ms:.0f}ms_cm_norm.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Best-epoch per-patch avg confusion matrix")
    parser.add_argument('--root', required=True)
    parser.add_argument('--subject', nargs='+', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_dir', default='./EEG(500Hz)_53ch')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--time_bin', type=int, default=16)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root, "avg_cm")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": 0,
        "shuffle": False,
        "sampling_rate": 500,
        "start_time_ms": -200,
        "data_ext": "npy",
        "window_type": "fixed",
        "time_bin": args.time_bin,
        "file_chunk_type": "subject",
        "normalize_method": "zscore",
        "patch_idx": None,
        "stride": args.stride,
        "save_dir": args.root,
        "n_classes": args.n_classes,
        "is_label_null": True,
        "skip_pred": False,
        "metrics": ["acc", "bal_acc"],
        "ckpt_dir": args.root,
        "test_files": None,
        "patch_size": args.patch_size,
    }

    subject_files_map = discover_all_subjects(config["data_dir"])

    # Model init
    first_subject = sorted(subject_files_map.keys())[0]
    first_files = subject_files_map[first_subject]
    config["test_files"] = first_files
    input_len, input_ch = COMBDataset(
        config=config, filepath=first_files)._get_sample_info()
    num_patches = (input_len - args.time_bin) // args.stride + 1

    model = EEGNet(n_channels=input_ch, n_timepoints=args.patch_size,
                   n_classes=args.n_classes).cuda(0)

    # Patch time info
    patch_times = {}
    for p_idx in range(num_patches):
        t_ms, _ = get_patch_time_ms_stride(
            p_idx, args.time_bin, args.stride, 500)
        patch_times[p_idx] = t_ms

    # Discover subjects
    subjects = sorted([d for d in os.listdir(args.root)
                       if d.startswith("sub-") and
                       os.path.isdir(os.path.join(args.root, d))])
    if args.subject:
        subjects = [s for s in subjects if s in args.subject]

    print(f"[*] Subjects: {len(subjects)}, patches: {num_patches}")

    # Per-patch prediction collection
    patch_preds = defaultdict(list)   # patch_idx -> list of preds arrays
    patch_labels = defaultdict(list)  # patch_idx -> list of labels arrays

    for subj in tqdm(subjects, desc="Subjects"):
        csv_path = os.path.join(args.root, subj, "combined_analysis",
                                f"{subj}_all_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [Skip] {subj}: no CSV")
            continue

        if subj not in subject_files_map:
            print(f"  [Skip] {subj}: no data files")
            continue

        df = pd.read_csv(csv_path)
        best_entries = extract_best_epoch_diag(df)

        subject_files = subject_files_map[subj]
        config["test_files"] = subject_files

        _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
            config=config, filepath=subject_files,
            val_size=0.2, test_size=0.2, random_state=42)

        subject_dir = os.path.join(args.root, subj)
        shared_cache = {}

        for train_patch_idx, epoch, ckpt_name in best_entries:
            train_patch_idx = int(train_patch_idx)
            ckpt_path = find_ckpt_path(subject_dir, ckpt_name)
            if ckpt_path is None:
                continue

            loaded = load_model(model, ckpt_path)
            if loaded is None:
                continue

            preloaded = preload_patch_tensors(
                config, subject_files, test_trial_indices,
                train_patch_idx, shared_cache)

            preds, labels = predict(model, preloaded, args.batch_size, args.patch_size)

            patch_preds[train_patch_idx].append(preds)
            patch_labels[train_patch_idx].append(labels)

            del preloaded

        del shared_cache
        torch.cuda.empty_cache()
        tqdm.write(f"  [Done] {subj}: {len(best_entries)} patches")

    # Generate per-patch CMs
    print(f"\n[*] Generating per-patch CMs...")
    label_names = np.arange(args.n_classes)
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

        plot_cm(cm, cm_norm, acc, bal_acc, p_idx, time_ms, n_subj,
                args.n_classes, output_dir)

        patch_results.append({
            'patch_idx': p_idx, 'time_ms': time_ms,
            'acc': acc, 'bal_acc': bal_acc, 'n_subjects': n_subj
        })

        all_preds_global.extend(preds_all)
        all_labels_global.extend(labels_all)

    # Overall CM (all patches combined)
    all_preds_global = np.array(all_preds_global)
    all_labels_global = np.array(all_labels_global)
    cm_total = confusion_matrix(all_labels_global, all_preds_global, labels=label_names)
    row_sums = cm_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_total_norm = cm_total.astype(float) / row_sums * 100

    acc_total = accuracy_score(all_labels_global, all_preds_global) * 100
    bal_acc_total = balanced_accuracy_score(all_labels_global, all_preds_global) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_total_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=[f"Class_{i}" for i in label_names],
                yticklabels=[f"Class_{i}" for i in label_names])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"All Patches Best-Epoch CM (n={len(subjects)})\n"
                 f"Acc={acc_total:.2f}%, Bal_Acc={bal_acc_total:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_patches_overall_cm_norm.png"), dpi=150)
    plt.close()

    # Save summary CSV
    df_results = pd.DataFrame(patch_results)
    df_results.to_csv(os.path.join(output_dir, "per_patch_cm_summary.csv"), index=False)

    print(f"\n[*] Overall: acc={acc_total:.2f}%, bal_acc={bal_acc_total:.2f}%")
    print(f"[*] Generated {len(patch_results)} per-patch CMs + 1 overall CM")
    print(f"\n[!] Done. Output: {output_dir}")


if __name__ == '__main__':
    main()
