"""
Subject별 CSV에서 epoch별 TGM 그래프 생성.
utils_infer.Visualizer.plot_tgm 활용.

Usage:
  # 전체 subject
  python plot_tgm_per_epoch.py --root EEGPT/within_total/500Hz_t16_s4

  # 특정 subject만
  python plot_tgm_per_epoch.py --root EEGPT/within_total/500Hz_t16_s4 --subject sub-22 sub-23

  # 특정 epoch만
  python plot_tgm_per_epoch.py --root EEGPT/within_total/500Hz_t16_s4 --epoch 9 19
"""
import argparse
import os
import pandas as pd
from utils_infer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Plot per-epoch TGM from summary CSVs")
    parser.add_argument('--root', required=True,
                        help="Root dir (e.g. EEGPT/within_total/500Hz_t16_s4)")
    parser.add_argument('--subject', nargs='+', default=None,
                        help="Filter subjects (e.g. sub-22 sub-23)")
    parser.add_argument('--epoch', type=int, nargs='+', default=None,
                        help="Filter epochs (e.g. 9 19)")
    parser.add_argument('--metrics', nargs='+',
                        default=['test_acc', 'test_bal_acc'],
                        help="Metrics to plot")
    parser.add_argument('--n_classes', type=int, default=6)
    args = parser.parse_args()

    config = {
        "n_classes": args.n_classes,
        "save_dir": args.root,
        "time_bin": 16,
        "stride": 4,
        "sampling_rate": 500,
    }
    vis = Visualizer(config)

    # Discover subject CSVs
    subject_dirs = sorted([
        d for d in os.listdir(args.root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(args.root, d))
    ])

    if args.subject:
        subject_dirs = [d for d in subject_dirs if d in args.subject]

    print(f"[*] Subjects: {len(subject_dirs)}")

    for subj in subject_dirs:
        csv_path = os.path.join(args.root, subj, "combined_analysis",
                                f"{subj}_all_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [Skip] {subj}: no CSV")
            continue

        df = pd.read_csv(csv_path)
        epochs = sorted(df['epoch'].unique())

        if args.epoch:
            epochs = [e for e in epochs if e in args.epoch]

        print(f"\n[*] {subj}: {len(df)} rows, {len(epochs)} epochs")

        ep_dir = os.path.join(args.root, subj, "combined_analysis")
        os.makedirs(ep_dir, exist_ok=True)

        for ep in epochs:
            df_ep = df[df['epoch'] == ep]

            ckpt_label = f"{subj}_epoch{ep:02d}"

            for m in args.metrics:
                vis.plot_tgm(
                    ckpt=ckpt_label,
                    df=df_ep,
                    metric_key=m,
                    save_dir=ep_dir,
                    title=f"{subj} Epoch {ep} TGM ({m})",
                    filename_prefix=f"{subj}_ep{ep:02d}_tgm",
                )

            print(f"  epoch {ep:02d} -> {ep_dir}")

    print("\n[!] Done.")


if __name__ == '__main__':
    main()
