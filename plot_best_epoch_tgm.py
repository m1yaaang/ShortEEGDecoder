"""
Subject별 best-epoch TGM 생성.
각 train_patch에서 diagonal accuracy가 가장 높은 epoch를 선택,
해당 epoch의 전체 test patch(0~108) 데이터로 109×109 TGM 생성.

Usage:
  python plot_best_epoch_tgm.py --root EEGNet/within_logs/500Hz_t16_s4
  python plot_best_epoch_tgm.py --root EEGPT/within_total/500Hz_t16_s4
  python plot_best_epoch_tgm.py --root EEGNet/within_logs/500Hz_t16_s4 --subject sub-22 sub-70
  python plot_best_epoch_tgm.py --root EEGNet/within_logs/500Hz_t16_s4 --cross
"""
import argparse
import os
import pandas as pd
from utils_infer import Visualizer


def extract_best_epoch(df):
    """각 train_patch별 diagonal acc 최고 epoch 선택 → 전체 test patch 반환."""
    diag = df[df['train_patch_idx'] == df['test_patch_idx']]
    best = diag.loc[diag.groupby('train_patch_idx')['test_acc'].idxmax()]
    keys = best[['train_patch_idx', 'epoch']].values.tolist()
    parts = []
    for tp, ep in keys:
        parts.append(df[(df['train_patch_idx'] == tp) & (df['epoch'] == ep)])
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Best-epoch TGM per subject")
    parser.add_argument('--root', required=True)
    parser.add_argument('--subject', nargs='+', default=None)
    parser.add_argument('--cross', action='store_true',
                        help='전체 subject 평균 best-epoch TGM도 생성')
    parser.add_argument('--metrics', nargs='+', default=['test_acc', 'test_bal_acc'])
    parser.add_argument('--n_classes', type=int, default=6)
    args = parser.parse_args()

    config = {"n_classes": args.n_classes, "save_dir": args.root,
              "time_bin": 16, "stride": 4, "sampling_rate": 500}
    vis = Visualizer(config)

    all_best_dfs = []

    for entry in sorted(os.listdir(args.root)):
        if not entry.startswith("sub-") or not os.path.isdir(os.path.join(args.root, entry)):
            continue
        if args.subject and entry not in args.subject:
            continue
        csv_path = os.path.join(args.root, entry, "combined_analysis",
                                f"{entry}_all_summary.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df_best = extract_best_epoch(df)

        save_dir = os.path.join(args.root, entry, "combined_analysis")
        df_best.to_csv(os.path.join(save_dir, f"{entry}_best_epoch_summary.csv"),
                       index=False)

        for m in args.metrics:
            vis.plot_tgm(
                ckpt=entry, df=df_best, metric_key=m, save_dir=save_dir,
                title=f"{entry} Best-Epoch TGM ({m})",
                filename_prefix=f"{entry}_best_epoch_tgm")

        print(f"[*] {entry}: {len(df_best)} rows -> {save_dir}")
        all_best_dfs.append(df_best)

    if args.cross and all_best_dfs:
        df_all = pd.concat(all_best_dfs, ignore_index=True)
        n_subj = df_all['subject'].nunique()
        df_avg = df_all.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first', 'test_time_ms': 'first',
            'test_acc': 'mean', 'test_bal_acc': 'mean',
        }).reset_index()

        cross_dir = os.path.join(args.root, "cross_subject_tgm")
        df_all.to_csv(os.path.join(cross_dir, "all_subjects_best_epoch_summary.csv"),
                      index=False)
        df_avg.to_csv(os.path.join(cross_dir, "all_subjects_best_epoch_avg.csv"),
                      index=False)
        print(f"  CSV saved: all_subjects_best_epoch_summary.csv ({len(df_all)} rows)")
        print(f"  CSV saved: all_subjects_best_epoch_avg.csv ({len(df_avg)} rows)")
        for m in args.metrics:
            vis.plot_tgm(
                ckpt="overall", df=df_avg, metric_key=m, save_dir=cross_dir,
                title=f"All Subjects Best-Epoch Avg ({m}, n={n_subj})",
                filename_prefix="all_subjects_best_epoch_tgm")
        print(f"\n[*] Cross-subject best-epoch avg (n={n_subj}) -> {cross_dir}")

    print("\n[!] Done.")


if __name__ == '__main__':
    main()
