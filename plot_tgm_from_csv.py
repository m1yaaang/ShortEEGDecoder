"""
Summary CSV로부터 TGM (Temporal Generalization Matrix) 시각화.

Usage:
  # 단일 CSV → TGM + diagonal plot
  python plot_tgm_from_csv.py --csv EEGPT/within_total/500Hz_t16_s4/cross_subject_analysis/all_subjects_all_epochs_summary.csv

  # 출력 디렉토리 지정
  python plot_tgm_from_csv.py --csv path/to/summary.csv --output_dir ./my_plots

  # 특정 subject만
  python plot_tgm_from_csv.py --csv path/to/summary.csv --subject sub-22

  # 특정 epoch만
  python plot_tgm_from_csv.py --csv path/to/summary.csv --epoch 9
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class TGMPlotter:
    """Summary CSV 기반 TGM / Diagonal Accuracy 시각화."""

    def __init__(self, n_classes=6):
        self.n_classes = n_classes

    # ----------------------------------------------------------
    # Core: TGM heatmap
    # ----------------------------------------------------------
    def plot_tgm(self, df, metric_key='test_acc', save_dir='.',
                 title=None, filename_prefix='tgm'):
        os.makedirs(save_dir, exist_ok=True)

        tgm_matrix = df.pivot_table(
            index='train_patch_idx',
            columns='test_patch_idx',
            values=metric_key,
            aggfunc='mean',
        )
        tgm_matrix.sort_index(ascending=False, inplace=True)

        train_indices = tgm_matrix.index.tolist()
        test_indices = tgm_matrix.columns.tolist()

        # Labels with ms
        if 'train_time_ms' in df.columns:
            tmap = (df[['train_patch_idx', 'train_time_ms']]
                    .drop_duplicates().set_index('train_patch_idx'))
            train_labels = [f"P{i} ({tmap.loc[i,'train_time_ms']:.0f}ms)"
                            for i in train_indices]
        else:
            train_labels = [f"P{i}" for i in train_indices]

        if 'test_time_ms' in df.columns:
            tmap = (df[['test_patch_idx', 'test_time_ms']]
                    .drop_duplicates().set_index('test_patch_idx'))
            test_labels = [f"P{i} ({tmap.loc[i,'test_time_ms']:.0f}ms)"
                           for i in test_indices]
        else:
            test_labels = [f"P{i}" for i in test_indices]

        fig, ax = plt.subplots(figsize=(15, 13))
        sns.heatmap(
            tgm_matrix.values, annot=False, cmap='viridis',
            xticklabels=test_labels, yticklabels=train_labels,
            cbar_kws={'label': metric_key}, ax=ax,
        )

        stride = max(1, len(test_labels) // 12)
        ax.set_xticks(np.arange(len(test_labels))[::stride] + 0.5)
        ax.set_xticklabels([test_labels[i] for i in range(0, len(test_labels), stride)],
                           rotation=45, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(train_labels))[::stride] + 0.5)
        ax.set_yticklabels([train_labels[i] for i in range(0, len(train_labels), stride)],
                           rotation=0, fontsize=9)

        ax.set_title(title or f"TGM ({metric_key})", fontsize=16, pad=15)
        ax.set_xlabel("Test Time", fontsize=13)
        ax.set_ylabel("Train Time", fontsize=13)

        # Diagonal line
        ax.plot([0, len(test_labels)], [len(train_labels), 0],
                color='red', ls='--', lw=1.5, alpha=0.7)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{filename_prefix}_{metric_key}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  [TGM] {path}")

    # ----------------------------------------------------------
    # Core: Diagonal accuracy line plot
    # ----------------------------------------------------------
    def plot_diagonal(self, df, metric_key='test_acc', save_dir='.',
                      title=None, filename_prefix='diagonal'):
        os.makedirs(save_dir, exist_ok=True)

        df_diag = df[df['train_patch_idx'] == df['test_patch_idx']].copy()
        if df_diag.empty:
            print(f"  [Skip] No diagonal data for {metric_key}")
            return

        df_avg = df_diag.groupby('train_patch_idx').agg({
            'train_time_ms': 'first',
            metric_key: 'mean',
        }).reset_index().sort_values('train_patch_idx')

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_avg['train_time_ms'], df_avg[metric_key],
                lw=1.5, color='steelblue', label='epoch avg')
        mean_val = df_avg[metric_key].mean()
        ax.axhline(y=mean_val, color='red', ls='--', lw=1, alpha=0.7,
                    label=f"mean={mean_val:.2f}%")
        chance = 100.0 / self.n_classes
        ax.axhline(y=chance, color='gray', ls=':', lw=1, alpha=0.5,
                    label=f"chance={chance:.1f}%")

        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel(metric_key, fontsize=12)
        ax.set_title(title or f"Diagonal {metric_key}", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_dir, f"{filename_prefix}_{metric_key}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  [Diag] {path}")

    # ----------------------------------------------------------
    # Per-subject plots
    # ----------------------------------------------------------
    def plot_per_subject(self, df, save_dir, metrics=('test_acc', 'test_bal_acc')):
        subjects = sorted(df['subject'].unique())
        for subj in subjects:
            df_s = df[df['subject'] == subj]
            subj_dir = os.path.join(save_dir, subj)
            print(f"\n[*] {subj} ({len(df_s)} rows)")

            is_full = len(df_s['test_patch_idx'].unique()) > 1 or \
                      (df_s['train_patch_idx'] != df_s['test_patch_idx']).any()

            for m in metrics:
                if is_full:
                    self.plot_tgm(df_s, metric_key=m, save_dir=subj_dir,
                                  title=f"{subj} - TGM ({m}, epoch avg)",
                                  filename_prefix=f"{subj}_tgm")
                self.plot_diagonal(df_s, metric_key=m, save_dir=subj_dir,
                                   title=f"{subj} - Diagonal ({m})",
                                   filename_prefix=f"{subj}_diagonal")

    # ----------------------------------------------------------
    # Per-epoch cross-subject averaged TGM
    # ----------------------------------------------------------
    def plot_per_epoch(self, df, save_dir, metrics=('test_acc', 'test_bal_acc')):
        epochs = sorted(df['epoch'].unique())
        print(f"\n[*] Per-epoch TGM for {len(epochs)} epochs")

        for ep in epochs:
            df_ep = df[df['epoch'] == ep]
            n_subj = df_ep['subject'].nunique()
            label = f"epoch_{ep:02d}"

            df_avg = df_ep.groupby(['train_patch_idx', 'test_patch_idx']).agg({
                'train_time_ms': 'first',
                'test_time_ms': 'first',
                'test_acc': 'mean',
                'test_bal_acc': 'mean',
            }).reset_index()

            ep_dir = os.path.join(save_dir, label)
            for m in metrics:
                self.plot_tgm(df_avg, metric_key=m, save_dir=ep_dir,
                              title=f"Epoch {ep} Cross-Subject Avg ({m}, n={n_subj})",
                              filename_prefix=f"{label}_cross_avg")
                self.plot_diagonal(df_avg, metric_key=m, save_dir=ep_dir,
                                   title=f"Epoch {ep} Cross-Subject Diagonal ({m}, n={n_subj})",
                                   filename_prefix=f"{label}_cross_diag")

    # ----------------------------------------------------------
    # Overall averaged
    # ----------------------------------------------------------
    def plot_overall(self, df, save_dir, metrics=('test_acc', 'test_bal_acc')):
        n_subj = df['subject'].nunique()
        df_avg = df.groupby(['train_patch_idx', 'test_patch_idx']).agg({
            'train_time_ms': 'first',
            'test_time_ms': 'first',
            'test_acc': 'mean',
            'test_bal_acc': 'mean',
        }).reset_index()

        print(f"\n[*] Overall avg TGM ({n_subj} subjects, all epochs)")
        for m in metrics:
            self.plot_tgm(df_avg, metric_key=m, save_dir=save_dir,
                          title=f"All Subjects/Epochs Avg ({m}, n={n_subj})",
                          filename_prefix="overall_avg_tgm")
            self.plot_diagonal(df_avg, metric_key=m, save_dir=save_dir,
                               title=f"All Subjects/Epochs Diagonal ({m}, n={n_subj})",
                               filename_prefix="overall_avg_diagonal")


def main():
    parser = argparse.ArgumentParser(description="Plot TGM from summary CSV")
    parser.add_argument('--csv', required=True, help="Path to summary CSV")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory (default: same as CSV)")
    parser.add_argument('--subject', nargs='*', default=None,
                        help="Filter specific subjects (e.g. sub-22 sub-23)")
    parser.add_argument('--epoch', type=int, nargs='*', default=None,
                        help="Filter specific epochs (e.g. 9 19)")
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--skip_per_epoch', action='store_true',
                        help="Skip per-epoch plots (faster)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[*] Loaded {args.csv}: {len(df)} rows, "
          f"{df['subject'].nunique()} subjects, "
          f"{df['epoch'].nunique()} epochs")

    if args.subject:
        df = df[df['subject'].isin(args.subject)]
        print(f"  Filtered to subjects: {args.subject} → {len(df)} rows")
    if args.epoch:
        df = df[df['epoch'].isin(args.epoch)]
        print(f"  Filtered to epochs: {args.epoch} → {len(df)} rows")

    if df.empty:
        print("[!] No data after filtering.")
        return

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.csv), "tgm_plots")

    plotter = TGMPlotter(n_classes=args.n_classes)

    # 1. Per-subject
    plotter.plot_per_subject(df, save_dir=os.path.join(out_dir, "per_subject"))

    # 2. Per-epoch cross-subject (optional)
    if not args.skip_per_epoch:
        plotter.plot_per_epoch(df, save_dir=os.path.join(out_dir, "per_epoch"))

    # 3. Overall
    plotter.plot_overall(df, save_dir=out_dir)

    print(f"\n[!] Done. All plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
