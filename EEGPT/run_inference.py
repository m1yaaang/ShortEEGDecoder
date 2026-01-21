import torch
import os
import pickle
import numpy as np
from functools import partial
import tqdm
import pandas as pd
import wandb
from datetime import datetime
import matplotlib.pyplot as plt

from finetune_EEGPT_combine import (
    LitEEGPTCausal, 
    COMBLoader, 
)

from finetune_EEGPT_combine_LoRA import LitEEGPTCausal_LoRA

def load_model_from_checkpoint(ckpt_path, train_patch_idx= None, trained_structure=None, device='cuda'):
    print(f"[*] Loading model from {ckpt_path}...")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # 1. init model(same options as finetuning)
    if trained_structure is None:
        model = LitEEGPTCausal(
                        fixed_train_patch_idx=train_patch_idx,
                        load_path = None
                    )
    elif trained_structure == "LoRA":
        model = LitEEGPTCausal_LoRA(
                        fixed_train_patch_idx=train_patch_idx,
                        load_path = None
                    )

    # 2. load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 3. overload weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(device)
    model.eval()      # Dropout off and Fix BatchNorm
    return model


def predict_timebin_prob(models, test_loader, device, mask):
    """
    [기능] - 특정 타임빈만큼의 prob 계산(only short-term)
    배치 단위의 입력 데이터에 특정 시간 마스크(timebin)을 씌운 뒤,
    model에 통과시켜 각 클래스일 확률을 계산(단발용)
    
    [입력]
    - models(class or list):  
        - class : 학습된 모델 1개(멀티클래스)
        - list : 학습된 5개의 모델 리스트[Model_cls1, Model_cls2, Model_cls3] (바이너리클래스)
    - test_loader: 입력 EEG 데이터
    - device
    - mask(tensor): 테스트하고자 하는 시간대만 1이고 나머지는 0인 마스크 [Batch, Patch_Num]

    [출력]
    - prob(tensor): 각 클래스별 확률값 [Batch, 5]
    """
    if isinstance(models, list):
        for m in models: m.eval()
    else:
        models.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            x, y, _ = batch
            x = x.to(device)
            if mask.dim() == 2 and mask.shape[0] != x.shape[0]:
                # mask: [1, Patch] -> [Batch, Patch]
                batch_mask = mask.repeat(x.shape[0], 1).to(device)
            else:
                batch_mask = mask.to(device)


            # ---------------------------------------------------------
            # Case A: Binary Ensemble (모델 리스트인 경우)
            # ---------------------------------------------------------
            if isinstance(models, list):
                model_outputs = []
                for model in models:
                    # Forward
                    x_conv = model.chan_conv(x)
                    z = model.target_encoder(x_conv, model.chans_id.to(x))
                    if len(z.shape) == 4: z = z.flatten(2)
                    
                    # Masking
                    h, _ = model.pooler(z, mask=batch_mask)
                    logit = model.head(h)

                    # Logit -> Prob 변환
                    if logit.shape[-1] == 1:
                        prob = torch.sigmoid(logit) # [Batch, 1]
                    else:
                        prob = torch.softmax(logit, dim=-1)[:, 1].unsqueeze(1)
                    
                    model_outputs.append(prob)
                
                # 5개의 모델 결과를 옆으로 붙임: [Batch, 1] x 5 -> [Batch, 5]
                batch_prob = torch.cat(model_outputs, dim=1)

            # ---------------------------------------------------------
            # Case B: Multi-class (단일 모델인 경우)
            # ---------------------------------------------------------
            else:
                model = models
                x_conv = model.chan_conv(x)
                z = model.target_encoder(x_conv, model.chans_id.to(x))
                if len(z.shape) == 4: z = z.flatten(2)
                
                h, _ = model.pooler(z, mask=batch_mask)
                logit = model.head(h) # [Batch, 5]
                
                # Multi-class는 Softmax
                batch_prob = torch.softmax(logit, dim=-1)
            # 2. 결과 저장 (CPU로 이동하여 저장)
            all_probs.append(batch_prob.cpu())
            all_labels.append(y.cpu())

    # 3. 전체 배치를 하나의 텐서로 병합
    # total_probs: [Total_Data_Num, 5]
    # total_labels: [Total_Data_Num]
    total_probs = torch.cat(all_probs, dim=0)
    total_labels = torch.cat(all_labels, dim=0)

    return total_probs, total_labels

def save_prediction_to_csv(total_probs, total_labels, patch_indices=None, save_name="result.csv"):
    """
    [기능]
    - [Patch, Batch, Class] 형태로 쌓인 리스트를 하나의 CSV 파일로 병합하여 저장

    [입력]
    - total_probs: 확률값 리스트 (각 요소는 [Batch, num_classes] 텐서)
    - total_labels: 정답값 리스트 (각 요소는 [Batch] 텐서)
    - patch_indices: (옵션) 각 리스트 요소가 몇 번 패치인지 알려주는 리스트. 없으면 0, 1, 2... 순서대로 매김
    - save_name: 저장할 CSV 파일 경로

    [출력]
    - final_df: 저장된 DataFrame
    """

    all_dataframes = []

    # 1. 패치 단위로 반복 (P개)
    for i, (probs_tensor, label_tensor) in enumerate(zip(total_probs, total_labels)):

        # 텐서를 Numpy 배열로 변환 (CPU로 내리고 detach 필수)
        # probs_np: [Batch_Size, num_classes]
        probs_np = probs_tensor.detach().cpu().numpy()

        # label_np: [Batch_Size]
        label_np = label_tensor.detach().cpu().numpy().flatten()

        # Predicted Label (argmax)
        pred_np = np.argmax(probs_np, axis=1)

        # 2. 기본 DataFrame 생성 (확률값 먼저 넣기)
        # 컬럼 이름 자동 생성: Prob_Class0, Prob_Class1, ...
        num_classes = probs_np.shape[1]
        col_names = [f"Prob_Class{k}" for k in range(num_classes)]

        df_patch = pd.DataFrame(probs_np, columns=col_names)

        # 3. 추가 정보 컬럼 붙이기
        # Sample ID (각 패치 내에서의 샘플 인덱스)
        df_patch["Sample_ID"] = range(len(label_np))

        # Patch Num (현재 몇 번째 패치 데이터인지)
        current_patch_idx = patch_indices[i] if patch_indices else i
        df_patch["Patch_Num"] = current_patch_idx

        # Real Label (정답)
        df_patch["Real_Label"] = label_np

        # Predicted Label (예측값)
        df_patch["Pred_Label"] = pred_np

        # 리스트에 추가
        all_dataframes.append(df_patch)

    # 4. 모든 패치 데이터를 위아래로 합치기 (Concat)
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # 5. 컬럼 순서 예쁘게 정렬
    # 원하는 순서: [Sample_ID, Patch_Num, Prob_Class0..., Real_Label, Pred_Label]
    prob_cols = [c for c in final_df.columns if "Prob_" in c]
    cols = ["Sample_ID", "Patch_Num"] + prob_cols + ["Real_Label", "Pred_Label"]
    final_df = final_df[cols]

    # 6. CSV 파일로 저장
    final_df.to_csv(save_name, index=False)
    print(f"[*] CSV saved at: {save_name}")
    print(f"[*] Total Samples: {len(final_df)}")

    return final_df


def compute_metrics(total_probs, total_labels, patch_indices=None, num_classes=5, print_results=True):
    """
    [기능]
    - predict_timebin_prob의 출력을 받아서 각 패치별/전체 정확도 메트릭을 계산

    [입력]
    - total_probs: 확률값 리스트 (각 요소는 [Batch, num_classes] 텐서)
    - total_labels: 정답값 리스트 (각 요소는 [Batch] 텐서)
    - patch_indices: (옵션) 각 리스트 요소가 몇 번 패치인지 알려주는 리스트
    - num_classes: 클래스 개수 (기본값 5)
    - print_results: 결과 출력 여부

    [출력]
    - results_dict: {
        "per_patch": [{"patch_idx", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", ...}, ...],
        "overall": {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", ...}
      }
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report
    )

    per_patch_results = []
    all_preds = []
    all_labels = []

    # 1. 패치별 메트릭 계산
    for i, (probs_tensor, label_tensor) in enumerate(zip(total_probs, total_labels)):
        # 텐서 -> Numpy
        probs_np = probs_tensor.detach().cpu().numpy()
        labels_np = label_tensor.detach().cpu().numpy().flatten()
        preds_np = np.argmax(probs_np, axis=1)

        # 전체 집계용 저장
        all_preds.append(preds_np)
        all_labels.append(labels_np)

        # 패치 인덱스
        current_patch_idx = patch_indices[i] if patch_indices else i

        # 메트릭 계산
        acc = accuracy_score(labels_np, preds_np)
        bal_acc = balanced_accuracy_score(labels_np, preds_np)
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        f1_weighted = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0)

        patch_result = {
            "patch_idx": current_patch_idx,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "num_samples": len(labels_np)
        }
        per_patch_results.append(patch_result)

        if print_results:
            print(f"[Patch {current_patch_idx:2d}] Acc: {acc*100:.2f}% | Bal_Acc: {bal_acc*100:.2f}% | F1(macro): {f1_macro:.4f} | F1(weighted): {f1_weighted:.4f}")

    # 2. 전체 메트릭 계산 (모든 패치 통합)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    overall_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    overall_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    overall_result = {
        "accuracy": overall_acc,
        "balanced_accuracy": overall_bal_acc,
        "f1_macro": overall_f1_macro,
        "f1_weighted": overall_f1_weighted,
        "precision_macro": overall_precision_macro,
        "recall_macro": overall_recall_macro,
        "num_samples": len(all_labels)
    }

    if print_results:
        print("\n" + "="*60)
        print("[Overall Metrics]")
        print(f"  Accuracy:          {overall_acc*100:.2f}%")
        print(f"  Balanced Accuracy: {overall_bal_acc*100:.2f}%")
        print(f"  F1 (macro):        {overall_f1_macro:.4f}")
        print(f"  F1 (weighted):     {overall_f1_weighted:.4f}")
        print(f"  Precision (macro): {overall_precision_macro:.4f}")
        print(f"  Recall (macro):    {overall_recall_macro:.4f}")
        print(f"  Total Samples:     {len(all_labels)}")
        print("="*60)

    return {
        "per_patch": per_patch_results,
        "overall": overall_result
    }


def save_metrics_to_csv(metrics_dict, save_path="metrics_result.csv"):
    """
    [기능]
    - compute_metrics의 출력을 CSV 파일로 저장

    [입력]
    - metrics_dict: compute_metrics의 출력 딕셔너리
    - save_path: 저장 경로

    [출력]
    - df: 저장된 DataFrame
    """
    # Per-patch 결과를 DataFrame으로 변환
    df = pd.DataFrame(metrics_dict["per_patch"])

    # Overall 결과를 마지막 행에 추가 (confusion_matrix 제외)
    overall = metrics_dict["overall"].copy()
    overall.pop("confusion_matrix", None)  # confusion_matrix는 제외
    overall["patch_idx"] = "Overall"

    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    # CSV 저장
    df.to_csv(save_path, index=False)
    print(f"[*] Metrics CSV saved at: {save_path}")

    return df


def plot_1d_tgm(metrics_dict, train_patch_idx,
                sampling_rate=256, time_bin=16, start_time=-200,
                save_path="tgm_1d.png", metric_keys=None):
    """
    [기능]
    - 단일 모델(특정 시간대에서 학습)의 모든 테스트 시간대 성능을 1D로 시각화
    - Train Time 1개 x Test Time 전체

    [입력]
    - metrics_dict: compute_metrics()의 출력
    - train_patch_idx: 해당 모델이 학습된 패치 인덱스
    - sampling_rate: 샘플링 레이트 (Hz)
    - time_bin: 시간 빈 크기 (samples)
    - start_time: 시작 시간 (ms)
    - save_path: 저장 경로
    - metric_keys: 플롯할 메트릭 리스트 (기본값: ["accuracy", "balanced_accuracy", "f1_macro"])

    [출력]
    - fig: matplotlib figure 객체
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "balanced_accuracy", "f1_macro"]

    per_patch = metrics_dict["per_patch"]

    # 테스트 패치 인덱스 및 시간 계산
    test_indices = [p["patch_idx"] for p in per_patch]

    def patch_to_time_ms(patch_idx):
        return start_time + patch_idx * time_bin * 1000 / sampling_rate

    test_times = [patch_to_time_ms(idx) for idx in test_indices]
    train_time = patch_to_time_ms(train_patch_idx)
    train_time_end = patch_to_time_ms(train_patch_idx + 1)

    # Figure 생성: Line Plot + Heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    # ===== [상단] Line Plot =====
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, metric_key in enumerate(metric_keys):
        values = [p[metric_key] for p in per_patch]
        label = metric_key.replace("_", " ").title()
        ax1.plot(test_times, values, marker='o', markersize=4, linewidth=2,
                 label=label, color=colors[i % len(colors)])

    # Train Time 표시 (세로선)
    ax1.axvline(x=train_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Train Time ({train_time:.0f}~{train_time_end:.0f}ms)')
    ax1.axvspan(train_time, train_time_end, alpha=0.2, color='red')

    ax1.set_xlabel("Test Time (ms)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title(f"1D Temporal Generalization: Train Patch {train_patch_idx} ({train_time:.0f}~{train_time_end:.0f}ms)",
                  fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # ===== [하단] Heatmap (Accuracy만) =====
    ax2 = axes[1]
    acc_values = np.array([p["accuracy"] for p in per_patch]).reshape(1, -1)

    im = ax2.imshow(acc_values, aspect='auto', cmap='viridis', vmin=0, vmax=1.0)

    # X축 레이블 (시간)
    tick_interval = max(1, len(test_times) // 10)
    ax2.set_xticks(np.arange(len(test_times))[::tick_interval])
    ax2.set_xticklabels([f"{t:.0f}" for t in test_times[::tick_interval]])
    ax2.set_yticks([0])
    ax2.set_yticklabels([f"P{train_patch_idx}"])

    ax2.set_xlabel("Test Time (ms)", fontsize=11)
    ax2.set_title("Accuracy Heatmap", fontsize=11)

    # Train Time 표시 (Heatmap에서)
    train_idx_in_test = None
    for i, idx in enumerate(test_indices):
        if idx == train_patch_idx:
            train_idx_in_test = i
            break
    if train_idx_in_test is not None:
        ax2.axvline(x=train_idx_in_test, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[*] 1D TGM Plot saved at: {save_path}")

    return fig


def build_tgm_matrix(all_model_metrics, metric_key="accuracy"):
    """
    [기능]
    - 여러 모델(각기 다른 시간대에서 학습)의 메트릭을 모아 TGM Matrix 생성

    [입력]
    - all_model_metrics: dict 형태 {train_patch_idx: metrics_dict, ...}
        - metrics_dict는 compute_metrics()의 출력
    - metric_key: TGM에 사용할 메트릭 ("accuracy", "balanced_accuracy", "f1_macro", etc.)

    [출력]
    - tgm_matrix: np.ndarray [num_train_patches, num_test_patches]
    - train_indices: 학습 패치 인덱스 리스트
    - test_indices: 테스트 패치 인덱스 리스트
    """
    # 학습 패치 인덱스 정렬
    train_indices = sorted(all_model_metrics.keys())

    # 테스트 패치 인덱스는 첫 번째 모델의 per_patch에서 가져옴
    first_metrics = all_model_metrics[train_indices[0]]
    test_indices = [p["patch_idx"] for p in first_metrics["per_patch"]]

    # TGM Matrix 초기화
    num_train = len(train_indices)
    num_test = len(test_indices)
    tgm_matrix = np.zeros((num_train, num_test))

    # Matrix 채우기
    for i, train_idx in enumerate(train_indices):
        metrics = all_model_metrics[train_idx]
        for j, patch_result in enumerate(metrics["per_patch"]):
            tgm_matrix[i, j] = patch_result[metric_key]

    return tgm_matrix, train_indices, test_indices


def plot_tgm_heatmap(tgm_matrix, train_indices, test_indices,
                     sampling_rate=256, time_bin=16, start_time=-200,
                     metric_name="Accuracy", save_path="tgm_heatmap.png",
                     vmin=None, vmax=None, cmap="viridis"):
    """
    [기능]
    - TGM Matrix를 Heatmap으로 시각화

    [입력]
    - tgm_matrix: np.ndarray [num_train_patches, num_test_patches]
    - train_indices: 학습 패치 인덱스 리스트
    - test_indices: 테스트 패치 인덱스 리스트
    - sampling_rate: 샘플링 레이트 (Hz)
    - time_bin: 시간 빈 크기 (samples)
    - start_time: 시작 시간 (ms)
    - metric_name: 메트릭 이름 (제목용)
    - save_path: 저장 경로
    - vmin, vmax: 컬러맵 범위
    - cmap: 컬러맵 종류

    [출력]
    - fig: matplotlib figure 객체
    """
    import seaborn as sns

    # 시간 레이블 생성 (ms 단위)
    def patch_to_time_ms(patch_idx):
        return start_time + patch_idx * time_bin * 1000 / sampling_rate

    train_labels = [f"{patch_to_time_ms(idx):.0f}" for idx in train_indices]
    test_labels = [f"{patch_to_time_ms(idx):.0f}" for idx in test_indices]

    # 전체 시간 범위 계산
    train_time_start = patch_to_time_ms(min(train_indices))
    train_time_end = patch_to_time_ms(max(train_indices) + 1)
    test_time_start = patch_to_time_ms(min(test_indices))
    test_time_end = patch_to_time_ms(max(test_indices) + 1)

    # Figure 생성
    fig, ax = plt.subplots(figsize=(12, 10))

    # Heatmap 그리기
    sns.heatmap(
        tgm_matrix,
        annot=True if tgm_matrix.shape[0] <= 10 else False,  # 작은 matrix만 숫자 표시
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=test_labels,
        yticklabels=train_labels,
        ax=ax,
        cbar_kws={"label": metric_name}
    )

    # 대각선 표시 (Train Time == Test Time)
    ax.plot([0, min(len(test_indices), len(train_indices))],
            [0, min(len(test_indices), len(train_indices))],
            'r--', linewidth=2, alpha=0.7)

    ax.set_xlabel(f"Test Time (ms) [{test_time_start:.0f} ~ {test_time_end:.0f}ms]", fontsize=12)
    ax.set_ylabel(f"Train Time (ms) [{train_time_start:.0f} ~ {train_time_end:.0f}ms]", fontsize=12)
    ax.set_title(f"Temporal Generalization Matrix ({metric_name})\n"
                 f"Train: {train_time_start:.0f}~{train_time_end:.0f}ms | Test: {test_time_start:.0f}~{test_time_end:.0f}ms",
                 fontsize=14)

    # X축 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[*] TGM Heatmap saved at: {save_path}")

    return fig


def save_tgm_to_csv(tgm_matrix, train_indices, test_indices,
                    metric_name="accuracy", save_path="tgm_matrix.csv"):
    """
    [기능]
    - TGM Matrix를 CSV 파일로 저장

    [입력]
    - tgm_matrix: np.ndarray [num_train_patches, num_test_patches]
    - train_indices: 학습 패치 인덱스 리스트
    - test_indices: 테스트 패치 인덱스 리스트
    - metric_name: 메트릭 이름
    - save_path: 저장 경로

    [출력]
    - df: 저장된 DataFrame
    """
    # DataFrame 생성 (행: Train, 열: Test)
    df = pd.DataFrame(
        tgm_matrix,
        index=[f"Train_P{idx}" for idx in train_indices],
        columns=[f"Test_P{idx}" for idx in test_indices]
    )

    df.to_csv(save_path)
    print(f"[*] TGM Matrix CSV saved at: {save_path}")

    return df


def run_tgm_analysis(ckpt_paths, train_patch_indices, test_loader,
                     device="cuda", trained_structure=None, num_classes=5,
                     sampling_rate=256, time_bin=16, start_time=-200,
                     save_dir="./tgm_results"):
    """
    [기능]
    - 전체 TGM 분석 파이프라인 실행
    - 여러 체크포인트(각기 다른 시간대에서 학습)를 로드하고 모든 시간대에서 테스트

    [입력]
    - ckpt_paths: 체크포인트 경로 리스트 (학습 시간대 순서대로)
    - train_patch_indices: 각 체크포인트가 학습된 패치 인덱스 리스트
    - test_loader: 테스트 데이터 로더
    - device: 디바이스
    - trained_structure: 모델 구조 (None or "LoRA")
    - num_classes: 클래스 개수
    - sampling_rate, time_bin, start_time: 시간 계산용 파라미터
    - save_dir: 결과 저장 디렉토리

    [출력]
    - tgm_results: dict containing TGM matrix and metrics

    [사용 예시]
    >>> ckpt_paths = [
    ...     "logs/.../Patch0/checkpoints/best.ckpt",
    ...     "logs/.../Patch1/checkpoints/best.ckpt",
    ...     ...
    ... ]
    >>> train_patch_indices = [0, 1, 2, ...]  # 각 ckpt가 학습된 패치
    >>> results = run_tgm_analysis(ckpt_paths, train_patch_indices, test_loader)
    """
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)

    all_model_metrics = {}  # {train_patch_idx: metrics_dict}

    print("="*60)
    print("[TGM Analysis] Starting...")
    print(f"  - Number of models: {len(ckpt_paths)}")
    print(f"  - Train patches: {train_patch_indices}")
    print("="*60)

    # 1. 각 모델별로 추론 수행
    for ckpt_path, train_patch_idx in zip(ckpt_paths, train_patch_indices):
        print(f"\n[Model] Train Patch {train_patch_idx}: {ckpt_path}")

        # 모델 로드
        model = load_model_from_checkpoint(
            ckpt_path,
            train_patch_idx=train_patch_idx,
            trained_structure=trained_structure,
            device=device
        )

        # 모든 테스트 패치에 대해 추론
        total_probs, total_labels = [], []
        num_patches = model.target_encoder.patch_embed.num_patches

        for test_patch_idx in range(model.total_blocks):
            # 마스크 생성
            mask = torch.zeros((1, num_patches))
            if test_patch_idx < num_patches:
                mask[0, test_patch_idx] = 1.0
            else:
                print(f"  [Warning] Test patch {test_patch_idx} >= num_patches {num_patches}, skipping")
                continue

            # 추론
            probs, labels = predict_timebin_prob(model, test_loader, device, mask)
            total_probs.append(probs)
            total_labels.append(labels)

        # 메트릭 계산
        metrics = compute_metrics(
            total_probs, total_labels,
            patch_indices=list(range(len(total_probs))),
            num_classes=num_classes,
            print_results=False
        )

        all_model_metrics[train_patch_idx] = metrics

        print(f"  -> Overall Accuracy: {metrics['overall']['accuracy']*100:.2f}%")

    # 2. TGM Matrix 생성 (여러 메트릭)
    metrics_to_plot = ["accuracy", "balanced_accuracy", "f1_macro"]

    for metric_key in metrics_to_plot:
        tgm_matrix, train_indices, test_indices = build_tgm_matrix(
            all_model_metrics, metric_key=metric_key
        )

        # Heatmap 저장
        plot_tgm_heatmap(
            tgm_matrix, train_indices, test_indices,
            sampling_rate=sampling_rate, time_bin=time_bin, start_time=start_time,
            metric_name=metric_key.replace("_", " ").title(),
            save_path=os.path.join(save_dir, f"tgm_{metric_key}.png"),
            vmin=0, vmax=1.0 if "accuracy" in metric_key else None
        )

        # CSV 저장
        save_tgm_to_csv(
            tgm_matrix, train_indices, test_indices,
            metric_name=metric_key,
            save_path=os.path.join(save_dir, f"tgm_{metric_key}.csv")
        )

    # 3. 전체 결과 요약 저장
    summary_rows = []
    for train_idx, metrics in all_model_metrics.items():
        for patch_result in metrics["per_patch"]:
            summary_rows.append({
                "train_patch": train_idx,
                "test_patch": patch_result["patch_idx"],
                "accuracy": patch_result["accuracy"],
                "balanced_accuracy": patch_result["balanced_accuracy"],
                "f1_macro": patch_result["f1_macro"],
                "f1_weighted": patch_result["f1_weighted"]
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(save_dir, "tgm_full_results.csv"), index=False)

    print("\n" + "="*60)
    print(f"[TGM Analysis] Complete! Results saved to: {save_dir}")
    print("="*60)

    return {
        "all_model_metrics": all_model_metrics,
        "tgm_matrix": tgm_matrix,
        "train_indices": train_indices,
        "test_indices": test_indices
    }



if __name__ == "__main__":

    ''' Auto Loading Best Model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    '''

    # ==========================================
    # [설정] Configuration
    # ==========================================

    # CKPT_PATH = "./eegpt_combine/20251211_1531_patchsize=12_1/checkpoints/best-epoch=04-valid_loss=1.6087.ckpt"
    # CKPT_PATH = "logs/final_model_epoch100.ckpt"

    # 체크포인트 경로 리스트 (각각 다른 시간대(Patch)에서 학습된 모델들)
    CKPT_PATH = [
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P9_CAll/checkpoints/LORA_F1_P9_CAll-epoch=49-valid_loss=1.7860.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P10_CAll/checkpoints/LORA_F1_P10_CAll-epoch=49-valid_loss=1.7995.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P11_CAll/checkpoints/LORA_F1_P11_CAll-epoch=49-valid_loss=1.9003.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P12_CAll/checkpoints/LORA_F1_P12_CAll-epoch=49-valid_loss=1.8214.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P13_CAll/checkpoints/LORA_F1_P13_CAll-epoch=49-valid_loss=1.8155.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P14_CAll/checkpoints/LORA_F1_P14_CAll-epoch=49-valid_loss=1.8349.ckpt',
        '/home/myyu/Desktop/Link to myyu/EEGPT/downstream_combine3/logs/20260118_2054_LORA_F1_P15_CAll/checkpoints/LORA_F1_P15_CAll-epoch=49-valid_loss=1.8749.ckpt'
        ]   
    
    '''epoch39도 확인해보기 '''
    # 각 체크포인트가 학습된 패치 인덱스 (CKPT_PATH와 순서 동일)
    # TGM 분석 시 필요: Train Time(Y축)을 결정함
    TRAIN_PATCH_INDICES = [int(ckpt_path.split('_P')[1].split('_')[0]) for ckpt_path in CKPT_PATH]

    STRUCTURE = "LoRA"   # None or "LoRA"
    DATA_ROOT = "/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    TIME_BIN = 16           # 시간 빈 크기 (samples), 256Hz 기준 16 = 62.5ms
    NUM_CLASSES = 5         # 분류 클래스 수
    SAMPLING_RATE = 256     # 샘플링 레이트 (Hz)
    START_TIME = -200       # 시작 시간 (ms), stimulus onset 기준

    # 분석 모드 선택 (True/False)
    RUN_INDIVIDUAL_ANALYSIS = True   # 개별 모델 분석 (모델별 predictions, metrics CSV 저장)
    RUN_TGM_ANALYSIS = False          # TGM 분석 (Temporal Generalization Matrix)

    # ==========================================
    # [1단계] 테스트 데이터 준비
    # ==========================================
    print("[*] Preparing Test Data...")

    test_files = os.listdir(os.path.join(DATA_ROOT, "processed_test"))
    test_dataset = COMBLoader(
        os.path.join(DATA_ROOT, "processed_test"),
        test_files,
        is_train=False,
        time_bin=TIME_BIN
    )

    # Batch Size is influenced by the memory size of the GPU(can be bigger if needed)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False
    )

    data_length = test_dataset.total_blocks

    # ==========================================
    # [2단계] 개별 모델 분석 (Individual Model Analysis)
    # - 각 모델(특정 시간대에서 학습)을 모든 테스트 시간대에서 평가
    # - 결과: predictions CSV, metrics CSV
    # ==========================================
    if RUN_INDIVIDUAL_ANALYSIS:
        print("\n" + "="*60)
        print("[Individual Model Analysis] Starting...")
        print("="*60)

        experiment_name_list = []
        exp_id_list = []
        models_list = []

        for ckpt_path, train_patch_idx in zip(CKPT_PATH, TRAIN_PATCH_INDICES):  
            experiment_name = ckpt_path.split('/checkpoints/')[1].split('-epoch')[0]

            # 1. load Model
            model = load_model_from_checkpoint(ckpt_path, train_patch_idx=train_patch_idx, trained_structure=STRUCTURE, device=DEVICE)

            save_root = f"{ckpt_path.split('/checkpoints/')[0]}/analysis"
            os.makedirs(save_root, exist_ok=True)

            # [WANDB] 로깅 활성화 시 주석 해제
            run = wandb.init(
                project="infer_eegpt_combine_LoRA",
                name = experiment_name,
                job_type='Inference',
                tags=["inference", "heatmap","Test",'epoch49']
            )

            total_probs, total_labels = [], []
            for test_patch_idx in range(test_dataset.total_blocks):   # 0~15 patch idx
                
                print(f"\n[Inference] Experiment: {experiment_name} | Test Patch Index: {test_patch_idx}")

                # Create mask for the specific test patch index
                # num_patches: (channel_patches, seq_patches) 형태의 튜플
                # z shape: (B, seq_patches, embed_dim) -> pooler expects mask of (B, seq_patches)
                num_patches_tuple = model.target_encoder.patch_embed.num_patches
                num_seq_patches = num_patches_tuple[1] if isinstance(num_patches_tuple, tuple) else num_patches_tuple
                mask = torch.zeros((1, num_seq_patches))
                if test_patch_idx < num_seq_patches:
                    mask[0, test_patch_idx] = 1.0
                else:
                    print(f"[*] Warning: Test Patch Index {test_patch_idx} exceeds number of patches {num_seq_patches}. Skipping.")
                    continue

                # 2. Save Probabilities
                probs, labels = predict_timebin_prob(model, test_loader, DEVICE, mask)
                total_probs.append(probs)       # 모든 테스트에 대한 확률값 저장 [P, N, 5]
                total_labels.append(labels)     # 모든 테스트에 대한 레이블 저장 [P, N, 1]

            # CSV 저장: 각 테스트 패치별 예측 확률
            save_prediction_to_csv(total_probs, total_labels, patch_indices=range(test_dataset.total_blocks), save_name=f"{save_root}/predictions_{experiment_name}.csv")

            # 메트릭 계산 및 저장: Accuracy, Balanced Accuracy, F1 Score 등
            metrics = compute_metrics(total_probs, total_labels, num_classes=NUM_CLASSES)
            save_metrics_to_csv(metrics, save_path=f"{save_root}/metrics_{experiment_name}.csv")

            # Confusion matrix: only use the test patch that matches train_patch_idx
            from sklearn.metrics import confusion_matrix
            matched_probs = total_probs[train_patch_idx].detach().cpu().numpy()
            matched_labels = total_labels[train_patch_idx].detach().cpu().numpy().flatten()
            matched_preds = np.argmax(matched_probs, axis=1)
            matched_cm = confusion_matrix(matched_labels, matched_preds, labels=list(range(NUM_CLASSES)))

            # -------------------------------------------------------
            # [필수] 1D TGM 플롯: Train 1개 x Test 전체
            # - 해당 모델(특정 시간대에서 학습)이 모든 테스트 시간대에서 어떤 성능을 보이는지 시각화
            # -------------------------------------------------------
            tgm_1d_path = f"{save_root}/tgm_1d_{experiment_name}.png"
            plot_1d_tgm(
                metrics_dict=metrics,
                train_patch_idx=int(train_patch_idx),
                sampling_rate=SAMPLING_RATE,
                time_bin=TIME_BIN,
                start_time=START_TIME,
                save_path=tgm_1d_path,
                metric_keys=["accuracy", "balanced_accuracy", "f1_macro"]
            )

            # -------------------------------------------------------
            # [WANDB] 전체 로깅
            # -------------------------------------------------------
            if wandb.run is not None:
                # 1. Overall Metrics 로깅
                overall = metrics["overall"]
                wandb.log({
                    "Overall/Accuracy": overall["accuracy"],
                    "Overall/Balanced_Accuracy": overall["balanced_accuracy"],
                    "Overall/F1_Macro": overall["f1_macro"],
                    "Overall/F1_Weighted": overall["f1_weighted"],
                    "Overall/Precision_Macro": overall["precision_macro"],
                    "Overall/Recall_Macro": overall["recall_macro"],
                    "Overall/Num_Samples": overall["num_samples"],
                })

                # 2. Per-Patch Metrics를 Table로 로깅
                per_patch_df = pd.DataFrame(metrics["per_patch"])
                wandb.log({
                    "PerPatch/Metrics_Table": wandb.Table(dataframe=per_patch_df)
                })

                # 3. Per-Patch Accuracy Line Plot
                # 시간 계산
                def patch_to_time_ms(patch_idx):
                    return START_TIME + patch_idx * TIME_BIN * 1000 / SAMPLING_RATE

                per_patch_df["time_ms"] = per_patch_df["patch_idx"].apply(patch_to_time_ms)

                wandb.log({
                    "PerPatch/Accuracy_Line": wandb.plot.line(
                        wandb.Table(dataframe=per_patch_df),
                        "time_ms", "accuracy",
                        title="Accuracy over Test Time"
                    ),
                    "PerPatch/Balanced_Accuracy_Line": wandb.plot.line(
                        wandb.Table(dataframe=per_patch_df),
                        "time_ms", "balanced_accuracy",
                        title="Balanced Accuracy over Test Time"
                    ),
                    "PerPatch/F1_Macro_Line": wandb.plot.line(
                        wandb.Table(dataframe=per_patch_df),
                        "time_ms", "f1_macro",
                        title="F1 Macro over Test Time"
                    ),
                })

                # 4. Confusion Matrix 로깅 (이미지로 저장 후 로깅) - only matched patch (train == test)
                import seaborn as sns
                cm = matched_cm

                # Train/Test 시간 계산 (matched patch)
                train_time_start = START_TIME + train_patch_idx * TIME_BIN * 1000 / SAMPLING_RATE
                train_time_end = START_TIME + (train_patch_idx + 1) * TIME_BIN * 1000 / SAMPLING_RATE

                fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=[f"Class_{i}" for i in range(NUM_CLASSES)],
                            yticklabels=[f"Class_{i}" for i in range(NUM_CLASSES)])
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('True')
                ax_cm.set_title(f'Confusion Matrix (Matched Patch)\n'
                               f'Train & Test Time: {train_time_start:.0f}~{train_time_end:.0f}ms (Patch {train_patch_idx})')
                cm_path = f"{save_root}/confusion_matrix_{experiment_name}.png"
                plt.tight_layout()
                plt.savefig(cm_path, dpi=150)
                plt.close()
                wandb.log({
                    "Overall/Confusion_Matrix": wandb.Image(cm_path)
                })

                # 5. 1D TGM 이미지 로깅
                wandb.log({
                    "TGM_1D/Heatmap": wandb.Image(tgm_1d_path, caption=f"Train Patch {train_patch_idx}")
                })

                # 6. CSV 파일 Artifact로 저장
                artifact = wandb.Artifact(
                    name=f"inference_results_{experiment_name}",
                    type="inference"
                )
                artifact.add_file(f"{save_root}/predictions_{experiment_name}.csv")
                artifact.add_file(f"{save_root}/metrics_{experiment_name}.csv")
                artifact.add_file(tgm_1d_path)
                wandb.log_artifact(artifact)

                wandb.finish()

        print("\n[*] Individual Model Analysis Done!")

    # ==========================================
    # [3단계] TGM 분석 (Temporal Generalization Matrix)
    # - 여러 모델(각기 다른 시간대에서 학습)을 모든 시간대에서 테스트
    # - 결과: TGM Heatmap (Train Time x Test Time), CSV
    # - TGM[i,j] = 시간대 i에서 학습한 모델을 시간대 j에서 테스트한 성능
    # ==========================================
    if RUN_TGM_ANALYSIS:
        print("\n" + "="*60)
        print("[TGM Analysis] Temporal Generalization Matrix")
        print("="*60)

        # TGM 결과 저장 디렉토리
        tgm_save_dir = "./logs/TGM_Analysis"

        # [WANDB] TGM 분석용 로깅 활성화 시 주석 해제
        # run = wandb.init(
        #     project="infer_eegpt_combine_LoRA",
        #     name=f"{experiment_name}",
        #     job_type='TGM',
        #     tags=["TGM", "heatmap", "temporal_generalization"]
        # )

        tgm_results = run_tgm_analysis(
            ckpt_paths=CKPT_PATH,
            train_patch_indices=TRAIN_PATCH_INDICES,
            test_loader=test_loader,
            device=DEVICE,
            trained_structure=STRUCTURE,
            num_classes=NUM_CLASSES,
            sampling_rate=SAMPLING_RATE,
            time_bin=TIME_BIN,
            start_time=START_TIME,
            save_dir=tgm_save_dir
        )

        # [WANDB] TGM 결과 로깅
        if wandb.run is not None:
            # TGM Heatmap 이미지 로깅
            wandb.log({
                "TGM/Accuracy": wandb.Image(os.path.join(tgm_save_dir, "tgm_accuracy.png")),
                "TGM/Balanced_Accuracy": wandb.Image(os.path.join(tgm_save_dir, "tgm_balanced_accuracy.png")),
                "TGM/F1_Macro": wandb.Image(os.path.join(tgm_save_dir, "tgm_f1_macro.png")),
            })
            wandb.finish()

        print(f"\n[*] TGM Analysis results saved to: {tgm_save_dir}")

    print("\n[*] All Analysis Done!")