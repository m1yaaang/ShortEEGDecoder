"""
EEGNet subject별 전체 epoch 평균 TGM + 전체 subject 평균 TGM.

Usage:
  # 피험자별 epoch-avg TGM
  python plot_eegnet_tgm.py

  # 피험자별 + 전체 피험자 평균
  python plot_eegnet_tgm.py --cross

  # 특정 피험자만
  python plot_eegnet_tgm.py --subject sub-22 sub-23

  # root 변경
  python plot_eegnet_tgm.py --root EEGNet/within_logs/500Hz_t16_s4
"""
import argparse
import os
import pandas as pd
from utils_infer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="EEGNet per-subject epoch-avg TGM")
    parser.add_argument('--root', default='EEGNet/within_logs/500Hz_t16_s4')
    parser.add_argument('--subject', nargs='+', default=None)
    parser.add_argument('--cross', action='store_true',
                        help='전체 subject 평균 TGM도 생성')
    parser.add_argument('--metrics', nargs='+', default=['test_acc', 'test_bal_acc'])
    parser.add_argument('--n_classes', type=int, default=6)
    args = parser.parse_args()

    config = {"n_classes": args.n_classes, "save_dir": args.root,
              "time_bin": 16, "stride": 4, "sampling_rate": 500}
    vis = Visualizer(config)

    # Load CSVs
    dfs = []
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
        dfs.append(df)

        # Per-subject epoch-avg TGM
        df_avg = df.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first', 'test_time_ms': 'first',
            'test_acc': 'mean', 'test_bal_acc': 'mean',
        }).reset_index()

        n_ep = df['epoch'].nunique()
        save_dir = os.path.join(args.root, entry, "combined_analysis")
        for m in args.metrics:
            vis.plot_tgm(
                ckpt=entry, df=df_avg, metric_key=m, save_dir=save_dir,
                title=f"{entry} Epoch-Avg TGM ({m}, {n_ep} epochs)",
                filename_prefix=f"{entry}_epoch_avg_tgm")
        print(f"[*] {entry} ({n_ep} epochs, {len(df)} rows) -> {save_dir}")

    # Cross-subject avg
    if args.cross and dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        n_subj = df_all['subject'].nunique()
        df_overall = df_all.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first', 'test_time_ms': 'first',
            'test_acc': 'mean', 'test_bal_acc': 'mean',
        }).reset_index()

        cross_dir = os.path.join(args.root, "cross_subject_tgm")
        for m in args.metrics:
            vis.plot_tgm(
                ckpt="overall", df=df_overall, metric_key=m,
                save_dir=cross_dir,
                title=f"All Subjects Epoch-Avg ({m}, n={n_subj})",
                filename_prefix="all_subjects_epoch_avg_tgm")
        print(f"\n[*] Cross-subject avg (n={n_subj}) -> {cross_dir}")

    print("\n[!] Done.")


if __name__ == '__main__':
    main()
