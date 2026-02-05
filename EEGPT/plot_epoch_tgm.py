"""
all_epochs_combined_summary.csv에서
동일한 epoch을 가진 ckpt끼리 모아서 epoch별 TGM을 생성하는 스크립트
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def extract_epoch(ckpt_name):
    """ckpt_name에서 epoch 번호 추출"""
    match = re.search(r'epoch=(\d+)', ckpt_name)
    return int(match.group(1)) if match else -1


def plot_tgm(df, metric_key, save_path, title):
    """TGM heatmap 생성"""
    tgm_matrix = df.pivot_table(
        index='train_patch_idx',
        columns='test_patch_idx',
        values=metric_key,
        aggfunc='mean'
    )
    tgm_matrix.sort_index(ascending=False, inplace=True)

    train_indices = tgm_matrix.index.tolist()
    test_indices = tgm_matrix.columns.tolist()

    # 라벨 생성
    if 'train_time_ms' in df.columns:
        train_map = df[['train_patch_idx', 'train_time_ms']].drop_duplicates().set_index('train_patch_idx')
        train_labels = [f"P{idx} ({train_map.loc[idx, 'train_time_ms']:.0f}ms)" for idx in train_indices]
    else:
        train_labels = [f"P{idx}" for idx in train_indices]

    if 'test_time_ms' in df.columns:
        test_map = df[['test_patch_idx', 'test_time_ms']].drop_duplicates().set_index('test_patch_idx')
        test_labels = [f"P{idx} ({test_map.loc[idx, 'test_time_ms']:.0f}ms)" for idx in test_indices]
    else:
        test_labels = [f"P{idx}" for idx in test_indices]

    plt.figure(figsize=(15, 13))
    sns.heatmap(
        tgm_matrix.values,
        annot=True if len(train_indices) <= 20 else False,
        fmt=".1f",
        cmap="viridis",
        xticklabels=test_labels,
        yticklabels=train_labels,
        cbar_kws={"label": metric_key}
    )

    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Test Time", fontsize=14)
    plt.ylabel("Train Time", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.plot([0, len(test_labels)], [len(train_labels),0],
             color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":

    csv_path = "./EEGPT/within/logs/combined_analysis/all_epochs_combined_summary.csv"
    output_dir = "./EEGPT/within/logs/combined_analysis/per_epoch"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['epoch'] = df['ckpt_name'].apply(extract_epoch)

    epochs = sorted(df['epoch'].unique())
    print(f"[*] Found {len(epochs)} epochs: {epochs}")
    print(f"[*] Total entries: {len(df)}")

    # -------------------------------------------------------
    # 1. Epoch별 TGM 생성
    # -------------------------------------------------------
    for epoch_num in epochs:
        df_epoch = df[df['epoch'] == epoch_num]
        epoch_label = f"epoch_{epoch_num:02d}"

        n_train_patches = df_epoch['train_patch_idx'].nunique()
        n_test_patches = df_epoch['test_patch_idx'].nunique()
        print(f"\n[*] {epoch_label}: {len(df_epoch)} entries, "
              f"train_patches={n_train_patches}, test_patches={n_test_patches}")

        # summary csv
        epoch_csv = os.path.join(output_dir, f"{epoch_label}_summary.csv")
        df_epoch.to_csv(epoch_csv, index=False)

        # TGM - acc
        plot_tgm(
            df_epoch,
            metric_key='test_acc',
            save_path=os.path.join(output_dir, f"{epoch_label}_TGM_test_acc.png"),
            title=f"Epoch {epoch_num} - TGM (test_acc)"
        )

        # TGM - bal_acc
        plot_tgm(
            df_epoch,
            metric_key='test_bal_acc',
            save_path=os.path.join(output_dir, f"{epoch_label}_TGM_test_bal_acc.png"),
            title=f"Epoch {epoch_num} - TGM (test_bal_acc)"
        )

    # -------------------------------------------------------
    # 2. 전체 평균 TGM (모든 epoch 평균)
    # -------------------------------------------------------
    print(f"\n[*] Generating mean TGM across all epochs...")

    plot_tgm(
        df,
        metric_key='test_acc',
        save_path=os.path.join(output_dir, "all_epochs_mean_TGM_test_acc.png"),
        title="All Epochs Mean - TGM (test_acc)"
    )
    plot_tgm(
        df,
        metric_key='test_bal_acc',
        save_path=os.path.join(output_dir, "all_epochs_mean_TGM_test_bal_acc.png"),
        title="All Epochs Mean - TGM (test_bal_acc)"
    )

    print(f"\n[!] Done. Output: {output_dir}")
