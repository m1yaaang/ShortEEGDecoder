"""
Permutation test for EEGNet within-subject decoding.
각 patch(train==test)의 best-epoch checkpoint으로 test data를 평가한 뒤,
label을 N번 셔플하여 null distribution을 만들고 p-value를 계산.

Usage:
  python permutation_test.py --subject sub-14
  python permutation_test.py --subject sub-14 --patch_size 16 --n_perm 500
  python permutation_test.py --subject sub-14 --data_dir ./EEG(500Hz)_COMB
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import recall_score

from utils_data import (discover_all_subjects, WithinSubjectDatasetTotal,
                        get_patch_time_ms_stride, COMBDataset)
from EEGNet.EEGNet_total import EEGNet


def parse_args():
    parser = argparse.ArgumentParser(description="Permutation test for EEGNet")
    parser.add_argument('--subject', required=True)
    parser.add_argument('--data_dir', default='./EEG(500Hz)_53ch')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--time_bin', type=int, default=16)
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--n_perm', type=int, default=1000,
                        help='number of permutations (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_model(model, ckpt_path):
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"[!] Corrupt checkpoint: {ckpt_path}\n    {e}")
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


def find_best_ckpt(subject_dir, best_entry):
    """best_epoch_summary에서 얻은 (train_patch_idx, ckpt_name) → ckpt 경로."""
    train_patch_idx, ckpt_name = best_entry
    for exp_name in os.listdir(subject_dir):
        ckpt_dir = os.path.join(subject_dir, exp_name, "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue
        ckpt_file = ckpt_name + ".ckpt"
        full_path = os.path.join(ckpt_dir, ckpt_file)
        if os.path.exists(full_path):
            return full_path
    return None


def predict_logits(model, X_t, mask_t, patch_size, batch_size):
    """Return predicted class indices."""
    n = X_t.shape[0]
    all_preds = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            x = X_t[s:e].cuda(0, non_blocking=True)
            m = mask_t[s:e].cuda(0, non_blocking=True)
            out = model(x, mask=m, patch_size=patch_size)
            all_preds.append(torch.argmax(out, dim=1).cpu())
    return torch.cat(all_preds).numpy()


def compute_bal_acc_no_null(preds, labels):
    """Class 0 제외 balanced accuracy (%)."""
    valid = labels != 0
    if valid.sum() == 0:
        return 0.0
    return 100 * recall_score(
        labels[valid], preds[valid],
        average='macro', zero_division=0.0,
        labels=np.unique(labels[valid]))


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    STRIDE = args.stride
    TIME_BIN = args.time_bin
    PATCH_SIZE = args.patch_size
    log_dir = f"./EEGNet/within_logs/500Hz_t{TIME_BIN}_s{STRIDE}_w{PATCH_SIZE}"
    subject_dir = os.path.join(log_dir, args.subject)

    # Load best epoch info
    csv_path = os.path.join(subject_dir, "combined_analysis",
                            f"{args.subject}_best_epoch_summary.csv")
    if not os.path.exists(csv_path):
        print(f"[!] No best_epoch_summary.csv: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    diag = df[df['train_patch_idx'] == df['test_patch_idx']]
    best_entries = diag.drop_duplicates('train_patch_idx')[
        ['train_patch_idx', 'ckpt_name']].values.tolist()

    config = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": 0,
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
        "n_classes": args.n_classes,
        "is_label_null": True,
        "skip_pred": False,
        "metrics": ["acc", "bal_acc"],
        "ckpt_dir": log_dir,
        "test_files": None,
        "patch_size": PATCH_SIZE,
    }

    # Discover subject files
    subject_files_map = discover_all_subjects(config["data_dir"])
    if args.subject not in subject_files_map:
        print(f"[!] Subject {args.subject} not found in {config['data_dir']}")
        sys.exit(1)

    subj_files = subject_files_map[args.subject]
    config["test_files"] = subj_files

    input_len, input_ch = COMBDataset(
        config=config, filepath=subj_files)._get_sample_info()
    num_patches = (input_len - TIME_BIN) // STRIDE + 1

    model = EEGNet(n_channels=input_ch, n_timepoints=PATCH_SIZE,
                   n_classes=args.n_classes).cuda(0)

    # Trial split (same as training)
    _, _, test_trial_indices = WithinSubjectDatasetTotal.split_trials(
        config=config, filepath=subj_files,
        val_size=0.2, test_size=0.2, random_state=42)

    # Patch time info
    patch_times = {}
    for p in range(num_patches):
        t_ms, _ = get_patch_time_ms_stride(p, TIME_BIN, STRIDE, 500)
        patch_times[p] = t_ms

    print(f"[*] Subject: {args.subject}")
    print(f"[*] Patches: {len(best_entries)}, Permutations: {args.n_perm}")
    print(f"[*] Data: {config['data_dir']} ({input_ch} ch)")

    # Run permutation test per patch
    results = []
    shared_cache = {}

    for train_patch_idx, ckpt_name in tqdm(best_entries, desc="Patches"):
        train_patch_idx = int(train_patch_idx)
        ckpt_path = find_best_ckpt(subject_dir, (train_patch_idx, ckpt_name))
        if ckpt_path is None:
            continue

        loaded = load_model(model, ckpt_path)
        if loaded is None:
            continue

        # Load test data for this patch
        cfg = config.copy()
        cfg["patch_idx"] = train_patch_idx
        ds = WithinSubjectDatasetTotal(
            config=cfg, filepath=subj_files,
            trial_indices=test_trial_indices, shared_cache=shared_cache)
        all_X, all_Y, all_mask = [], [], []
        for i in range(len(ds)):
            X, Y, stat, mask = ds[i]
            all_X.append(X); all_Y.append(Y); all_mask.append(mask)
        del ds

        X_t = torch.cat(all_X, dim=0)
        Y_t = torch.cat(all_Y, dim=0)
        mask_t = torch.cat(all_mask, dim=0)
        labels = Y_t.numpy()

        # Actual prediction
        preds = predict_logits(model, X_t, mask_t, PATCH_SIZE, args.batch_size)
        actual_acc = compute_bal_acc_no_null(preds, labels)

        # Permutation: shuffle labels
        perm_accs = np.zeros(args.n_perm)
        for i in range(args.n_perm):
            shuffled = rng.permutation(labels)
            perm_accs[i] = compute_bal_acc_no_null(preds, shuffled)

        # p-value (conservative: +1 to both numerator and denominator)
        p_value = (np.sum(perm_accs >= actual_acc) + 1) / (args.n_perm + 1)

        time_ms = patch_times.get(train_patch_idx, 0)
        results.append({
            'patch_idx': train_patch_idx,
            'time_ms': time_ms,
            'actual_bal_acc': actual_acc,
            'perm_mean': perm_accs.mean(),
            'perm_std': perm_accs.std(),
            'perm_95': np.percentile(perm_accs, 95),
            'p_value': p_value,
            'significant': p_value < args.alpha,
        })

        del X_t, Y_t, mask_t

    del shared_cache
    torch.cuda.empty_cache()

    # Save results
    df_res = pd.DataFrame(results).sort_values('patch_idx')
    output_dir = os.path.join(subject_dir, "combined_analysis")
    os.makedirs(output_dir, exist_ok=True)
    csv_out = os.path.join(output_dir,
                           f"{args.subject}_permutation_test.csv")
    df_res.to_csv(csv_out, index=False)

    n_sig = df_res['significant'].sum()
    print(f"\n[*] Significant patches: {n_sig}/{len(df_res)} "
          f"(alpha={args.alpha})")
    print(f"[*] Actual bal_acc range: "
          f"{df_res['actual_bal_acc'].min():.1f}% ~ "
          f"{df_res['actual_bal_acc'].max():.1f}%")
    print(f"[*] Permutation mean: {df_res['perm_mean'].mean():.1f}%")

    # Plot
    time_ms = df_res['time_ms'].values
    actual = df_res['actual_bal_acc'].values
    perm_95 = df_res['perm_95'].values
    sig = df_res['significant'].values
    chance = 100.0 / (args.n_classes - 1)  # class 0 제외 → 5 classes

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_ms, actual, '-', color='steelblue', linewidth=1.5,
            label='Actual bal_acc')
    ax.plot(time_ms, perm_95, '--', color='gray', linewidth=1.0,
            alpha=0.7, label=f'Permutation 95th percentile')
    ax.axhline(chance, color='lightgray', linestyle=':', linewidth=1.0,
               alpha=0.7, label=f'chance={chance:.1f}%')

    # Shade significant regions
    sig_times = time_ms[sig]
    sig_vals = actual[sig]
    if len(sig_times) > 0:
        ax.scatter(sig_times, sig_vals, color='red', s=15, zorder=5,
                   label=f'p<{args.alpha} (n={n_sig})')

    best_idx = np.argmax(actual)
    ax.plot(time_ms[best_idx], actual[best_idx], 'rv', markersize=10,
            zorder=6)
    ax.annotate(
        f'P{df_res.iloc[best_idx]["patch_idx"]:.0f} '
        f'({time_ms[best_idx]:.0f}ms, {actual[best_idx]:.1f}%)',
        xy=(time_ms[best_idx], actual[best_idx]),
        xytext=(10, -35 if actual[best_idx] > np.median(actual) else 15),
        textcoords='offset points', fontsize=9, color='red',
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title(f'{args.subject} - Permutation Test '
                 f'(n_perm={args.n_perm}, {n_sig}/{len(df_res)} sig.)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(False)
    plt.tight_layout()

    png_out = os.path.join(output_dir,
                           f"{args.subject}_permutation_test.png")
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[!] Done.")
    print(f"  CSV: {csv_out}")
    print(f"  Plot: {png_out}")


if __name__ == '__main__':
    main()
