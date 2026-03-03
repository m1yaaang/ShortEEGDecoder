"""
Subject별 CSV를 읽어 cross-subject epoch-averaged TGM 생성.

1) Per-subject: 전체 epoch 평균 TGM
2) Per-epoch: 전체 subject 평균 TGM
3) Overall: 전체 subject × 전체 epoch 평균 TGM

Usage:
  python plot_cross_subject_tgm.py --root EEGPT/within_total/500Hz_t16_s4
  python plot_cross_subject_tgm.py --root EEGNet/within_logs/500Hz_t16_s4
  python plot_cross_subject_tgm.py --root EEGPT/within_total/500Hz_t16_s4 --subject sub-22 sub-23
  python plot_cross_subject_tgm.py --root EEGPT/within_total/500Hz_t16_s4 --epoch 9 19
"""
import argparse
import os
import pandas as pd
from utils_infer import Visualizer


def load_all_subject_csvs(root_dir, target_subjects=None):
    dfs = []
    for entry in sorted(os.listdir(root_dir)):
        if not entry.startswith("sub-"):
            continue
        if target_subjects and entry not in target_subjects:
            continue
        csv_path = os.path.join(root_dir, entry, "combined_analysis",
                                f"{entry}_all_summary.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"  [Loaded] {entry}: {len(df)} rows, "
              f"epochs={sorted(df['epoch'].unique())}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-subject epoch-averaged TGM from summary CSVs")
    parser.add_argument('--root', required=True,
                        help="Root dir with sub-## folders")
    parser.add_argument('--subject', nargs='+', default=None)
    parser.add_argument('--epoch', type=int, nargs='+', default=None)
    parser.add_argument('--metrics', nargs='+',
                        default=['test_acc', 'test_bal_acc'])
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--output_dir', default=None,
                        help="Output dir (default: <root>/cross_subject_tgm)")
    args = parser.parse_args()

    config = {"n_classes": args.n_classes, "save_dir": args.root,
              "time_bin": 16, "stride": 4, "sampling_rate": 500}
    vis = Visualizer(config)

    print(f"[*] Loading CSVs from {args.root}")
    df_all = load_all_subject_csvs(args.root, args.subject)
    if df_all is None:
        print("[!] No CSVs found.")
        return

    if args.epoch:
        df_all = df_all[df_all['epoch'].isin(args.epoch)]

    subjects = sorted(df_all['subject'].unique())
    epochs = sorted(df_all['epoch'].unique())
    n_subj = len(subjects)
    print(f"\n[*] Total: {len(df_all)} rows, {n_subj} subjects, "
          f"{len(epochs)} epochs")

    out_dir = args.output_dir or os.path.join(args.root, "cross_subject_tgm")

    # ---------------------------------------------------------
    # 1. Per-epoch: cross-subject averaged TGM
    # ---------------------------------------------------------
    print(f"\n=== Per-Epoch Cross-Subject Avg TGM ===")
    per_epoch_dir = os.path.join(out_dir, "per_epoch")
    for ep in epochs:
        df_ep = df_all[df_all['epoch'] == ep]
        df_avg = df_ep.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first', 'test_time_ms': 'first',
            'test_acc': 'mean', 'test_bal_acc': 'mean',
        }).reset_index()

        n_s = df_ep['subject'].nunique()
        ep_dir = os.path.join(per_epoch_dir, f"epoch_{ep:02d}")
        for m in args.metrics:
            vis.plot_tgm(
                ckpt=f"epoch_{ep:02d}", df=df_avg, metric_key=m,
                save_dir=ep_dir,
                title=f"Epoch {ep} Cross-Subject Avg ({m}, n={n_s})",
                filename_prefix=f"epoch_{ep:02d}_cross_avg_tgm")
        print(f"  epoch {ep:02d} (n={n_s}) -> {ep_dir}")

    # ---------------------------------------------------------
    # 3. Overall: all subjects × all epochs averaged TGM
    # ---------------------------------------------------------
    print(f"\n=== Overall Avg TGM ===")
    df_overall = df_all.groupby(['train_patch_idx', 'test_patch_idx']).agg({
        'train_time_ms': 'first', 'test_time_ms': 'first',
        'test_acc': 'mean', 'test_bal_acc': 'mean',
    }).reset_index()

    for m in args.metrics:
        vis.plot_tgm(
            ckpt="overall", df=df_overall, metric_key=m,
            save_dir=out_dir,
            title=f"All Subjects/Epochs Avg ({m}, n={n_subj})",
            filename_prefix="overall_avg_tgm")

    print(f"\n[!] Done. Output: {out_dir}")


if __name__ == '__main__':
    main()
