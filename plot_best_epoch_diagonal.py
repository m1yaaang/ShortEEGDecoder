"""
Subject별 best-epoch diagonal accuracy 그래프 생성.
best_epoch_summary.csv에서 train_patch == test_patch인 대각선 accuracy를 시간축으로 플롯.

Usage:
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --subject sub-22 sub-70
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --cross
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --metric test_acc

사용법:                                                                                            
  # 전체 subject      
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4                          
                                                                                                     
  # 특정 subject만
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --subject sub-22 sub-70

  # cross-subject 평균 포함
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --cross

  # test_acc로 변경
  python plot_best_epoch_diagonal.py --root EEGNet/within_logs/500Hz_t16_s4 --metric test_acc

"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_diagonal(time_ms, values, metric, title, save_path, chance=16.7):
    best_idx = np.argmax(values)
    best_time = time_ms[best_idx]
    best_val = values[best_idx]
    best_patch = best_idx  # patch index == array index (sorted)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(time_ms, values, '-', color='steelblue', linewidth=1.5, label='best-epoch diag')
    ax.axhline(chance, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'chance={chance}%')
    ax.axvline(best_time, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    y_min, y_max = ax.get_ylim()
    if best_val > (y_min + y_max) / 2:
        xytext = (10, -35)
        va = 'top'
    else:
        xytext = (10, 15)
        va = 'bottom'

    ax.plot(best_time, best_val, 'rv', markersize=10, zorder=5)
    ax.annotate(f'P{best_patch} ({best_time:.0f}ms, {best_val:.1f}%)',
                xy=(best_time, best_val), xytext=xytext,
                textcoords='offset points', fontsize=9, color='red', fontweight='bold',
                va=va,
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Best-epoch diagonal accuracy plot")
    parser.add_argument('--root', required=True)
    parser.add_argument('--subject', nargs='+', default=None)
    parser.add_argument('--cross', action='store_true',
                        help='전체 subject 평균 diagonal도 생성')
    parser.add_argument('--metric', default='test_bal_acc')
    parser.add_argument('--n_classes', type=int, default=6)
    args = parser.parse_args()

    chance = (1.0 / args.n_classes) * 100

    subjects = sorted([d for d in os.listdir(args.root)
                        if d.startswith("sub-") and
                        os.path.isdir(os.path.join(args.root, d))])
    if args.subject:
        subjects = [s for s in subjects if s in args.subject]

    print(f"[*] Subjects: {len(subjects)}, metric: {args.metric}")

    all_diags = []

    for subj in subjects:
        csv_path = os.path.join(args.root, subj, "combined_analysis",
                                f"{subj}_best_epoch_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [Skip] {subj}: no CSV")
            continue

        df = pd.read_csv(csv_path)
        diag = df[df['train_patch_idx'] == df['test_patch_idx']].copy()
        diag = diag.sort_values('train_patch_idx').reset_index(drop=True)

        time_ms = diag['train_time_ms'].values
        values = diag[args.metric].values

        save_dir = os.path.join(args.root, subj, "combined_analysis")
        save_path = os.path.join(save_dir, f"{subj}_best_epoch_diagonal_{args.metric}.png")

        plot_diagonal(time_ms, values, args.metric,
                      f"{subj} - Best-Epoch Diagonal {args.metric}",
                      save_path, chance=chance)

        diag['subject'] = subj
        all_diags.append(diag)
        print(f"  {subj} -> {save_path}")

    if args.cross and all_diags:
        df_all = pd.concat(all_diags, ignore_index=True)
        n_subj = df_all['subject'].nunique()
        df_avg = df_all.groupby('train_patch_idx').agg({
            'train_time_ms': 'first',
            args.metric: 'mean',
        }).reset_index().sort_values('train_patch_idx')

        cross_dir = os.path.join(args.root, "cross_subject_tgm")
        os.makedirs(cross_dir, exist_ok=True)
        save_path = os.path.join(cross_dir,
                                 f"all_subjects_best_epoch_diagonal_{args.metric}.png")

        plot_diagonal(df_avg['train_time_ms'].values, df_avg[args.metric].values,
                      args.metric,
                      f"All Subjects Best-Epoch Diagonal {args.metric} (n={n_subj})",
                      save_path, chance=chance)
        print(f"\n[*] Cross-subject avg -> {save_path}")

    print("\n[!] Done.")


if __name__ == '__main__':
    main()
