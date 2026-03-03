"""
원본 코드:https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py#L224
-> 논문 코드 그대로임.

EEGNet within-subject training ver2 (per-patch, per-subject).

Changes from ver1:
  - Original EEGNet architecture (Lawhern et al. 2018)
    : Temporal Conv → BN → DepthwiseConv (spatial) → BN → ELU → AvgPool → Dropout
    : SeparableConv (Depthwise+Pointwise) → BN → ELU → AvgPool → Dropout
    : max_norm constraint: DepthwiseConv(1.0), FC(0.25)
    : F1=8, D=2, F2=16, dropout=0.5
  - AdamW + CosineAnnealingLR (lr scheduling)
  - batch_size default: 1024 -> 128
  - patience default: 5 -> 15
  - label_smoothing: 0.1
  - weight_decay: 1e-2

Usage:
  # 전체 subject, 기본 설정
  python EEGNet/EEGNet_total_ver2.py

  # 특정 subject만
  python EEGNet/EEGNet_total_ver2.py --subject sub-22 sub-70

  # stride, time_bin 변경
  python EEGNet/EEGNet_total_ver2.py --stride 4 --time_bin 16

  # batch_size, epochs, patience 변경
  python EEGNet/EEGNet_total_ver2.py --batch_size 256 --epochs 300 --patience 10

  # resume 비활성화 (처음부터 학습)
  python EEGNet/EEGNet_total_ver2.py --no_resume

  # GPU 지정
  python EEGNet/EEGNet_total_ver2.py --devices 0 1

  # 데이터 디렉토리 변경
  python EEGNet/EEGNet_total_ver2.py --data_dir ./EEG(500Hz)_53ch
"""

# python EEGNet/EEGNet_total_ver2.py --no_resume --devices 0 --patch_size 16

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, accuracy_score
from torch.autograd import Variable
import copy
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import logging
from tqdm import tqdm
from utils_data import WithinSubjectDatasetTotal, discover_all_subjects, torch_collate_fn
from utils_infer import InferenceManager
from utils_my import COMBDataset
from sklearn.model_selection import KFold
import datetime
import re
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime


class EEGNet(nn.Module):
    """Original EEGNet (Lawhern et al. 2018) - PyTorch implementation.

    Reference: https://github.com/vlawhern/arl-eegmodels

    Architecture:
      Block 1: Conv2D(temporal) → BN → DepthwiseConv2D(spatial) → BN → ELU → AvgPool → Dropout
      Block 2: SeparableConv2D → BN → ELU → AvgPool → Dropout
      Classifier: Flatten → FC

    Input: (B, C, T) where C=n_channels, T=n_timepoints
    """
    def __init__(self, n_channels=53, n_timepoints=64, n_classes=6,
                 F1=8, D=2, kern_length=None, dropout_rate=0.5):
        super().__init__()
        F2 = F1 * D

        if kern_length is None:
            kern_length = max(n_timepoints // 2, 4)

        # Adaptive pool sizes based on temporal length
        if n_timepoints >= 64:
            p1, p2 = 4, 8   # 원본과 동일 (total 32x)
        elif n_timepoints >= 32:
            p1, p2 = 4, 4   # total 16x
        else:
            p1, p2 = 2, 4   # total 8x (patch_size=16 → 16/8=2)

        # Block 1 - Temporal Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 1 - Depthwise Conv (Spatial)
        # groups=F1 → 각 temporal filter가 독립적으로 spatial filter 학습
        self.depthwise = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.pooling1 = nn.AvgPool2d((1, p1))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2 - Separable Conv (Depthwise + Pointwise)
        self.sep_depthwise = nn.Conv2d(F2, F2, (1, 16), padding='same', groups=F2, bias=False)
        self.sep_pointwise = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, p2))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self._calculate_fc_input_size(n_channels, n_timepoints)
        self.fc1 = nn.Linear(self.fc_input_size, n_classes)

    def _calculate_fc_input_size(self, n_channels, n_timepoints):
        """Forward pass를 통해 FC layer 입력 크기 계산"""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_timepoints)
            # Block 1
            x = self.batchnorm1(self.conv1(x))
            x = F.elu(self.batchnorm2(self.depthwise(x)))
            x = self.pooling1(x)
            # Block 2
            x = self.sep_pointwise(self.sep_depthwise(x))
            x = F.elu(self.batchnorm3(x))
            x = self.pooling2(x)
            self.fc_input_size = x.numel()

    def forward(self, x, mask=None, patch_size=None):
        # Mask & interpolate
        indices = torch.where(mask[0, 0] == 1)[0]
        x = x[:, :, indices]   # [B, C, time_bin]
        x = F.interpolate(x, size=patch_size, mode='linear', align_corners=True)

        # (B, C, T) → (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Block 1: Temporal → Spatial
        x = self.batchnorm1(self.conv1(x))          # (B, F1, C, T)
        x = F.elu(self.batchnorm2(self.depthwise(x)))  # (B, F2, 1, T)
        x = self.dropout1(self.pooling1(x))          # (B, F2, 1, T//p1)

        # Block 2: Separable Conv
        x = self.sep_pointwise(self.sep_depthwise(x))  # (B, F2, 1, T//p1)
        x = F.elu(self.batchnorm3(x))
        x = self.dropout2(self.pooling2(x))          # (B, F2, 1, T//(p1*p2))

        # Classifier
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x


class LitEEGNet(pl.LightningModule):
    """EEGNet Lightning Wrapper (ver2: Original architecture + AdamW + CosineAnnealing + LabelSmoothing)"""
    def __init__(self, n_channels, n_timepoints, n_classes, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = EEGNet(n_channels, n_timepoints, n_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.config = config

    def forward(self, x, mask=None, patch_size=None):
        return self.model(x, mask=mask, patch_size=patch_size)

    def training_step(self, batch, batch_idx):
        x, y, _, mask = batch
        logits = self(x, mask=mask, patch_size=self.config["patch_size"])
        loss = self.criterion(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, mask = batch
        logits = self(x, mask=mask, patch_size=self.config["patch_size"])
        loss = self.criterion(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        self.log('valid_acc', acc, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """max_norm constraint (원본 EEGNet과 동일)"""
        with torch.no_grad():
            # DepthwiseConv: max_norm(1.0)
            w = self.model.depthwise.weight
            norms = w.view(w.size(0), -1).norm(dim=1).clamp(min=1e-8)
            scale = (torch.clamp(norms, max=1.0) / norms).view(-1, 1, 1, 1)
            w.data *= scale

            # FC: max_norm(0.25)
            w = self.model.fc1.weight
            norms = w.norm(dim=1).clamp(min=1e-8)
            scale = (torch.clamp(norms, max=0.25) / norms).unsqueeze(1)
            w.data *= scale

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["num_epochs"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def evaluate(model, data_loader, params=["acc"]):
    """Multi-class classification용 evaluate 함수"""
    model.eval()
    results = []

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _, mask in data_loader:
            inputs = inputs.cuda(0)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(all_labels, all_preds))
        if param == "auc":
            # Multi-class AUC (one-vs-rest)
            try:
                results.append(roc_auc_score(all_labels, all_probs, multi_class='ovr'))
            except:
                results.append(0.0)
        if param == "recall":
            results.append(recall_score(all_labels, all_preds, average='macro'))
        if param == "precision":
            results.append(precision_score(all_labels, all_preds, average='macro', zero_division=0.0))
        if param == "fmeasure":
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0.0, labels=np.unique(all_labels))
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0.0, labels=np.unique(all_labels))
            if precision + recall > 0:
                results.append(2 * precision * recall / (precision + recall))
            else:
                results.append(0.0)

    model.train()
    return results, all_preds, all_labels


def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None or args.debug

def find_completed_patches(subject_dir):
    """subject_dir 내 완료된 patch index를 set으로 반환."""
    completed = set()
    if not os.path.exists(subject_dir):
        return completed
    for exp_name in os.listdir(subject_dir):
        match = re.search(r'_P(\d+)_t\d+_s\d+', exp_name)
        if not match:
            continue
        patch_idx = int(match.group(1))
        ckpt_dir = os.path.join(subject_dir, exp_name, "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue
        ckpts = [f for f in os.listdir(ckpt_dir)
                 if f.endswith('.ckpt') and f != 'last.ckpt']
        has_last = os.path.exists(os.path.join(ckpt_dir, 'last.ckpt'))
        if has_last or len(ckpts) > 0:
            completed.add(patch_idx)
    return completed


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="EEGNet within-subject training (ver2)")
    parser.add_argument('--data_dir', default='./EEG(500Hz)_53ch',
                        help='데이터 디렉토리 (default: ./EEG(500Hz)_53ch)')
    parser.add_argument('--subject', nargs='+', default=None,
                        help='학습할 subject (e.g. sub-22 sub-70). 미지정 시 전체')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--time_bin', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15,
                        help='EarlyStopping patience (default: 15)')
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--sampling_rate', type=int, default=500)
    parser.add_argument('--no_resume', action='store_true',
                        help='resume 비활성화 (처음부터 학습)')
    parser.add_argument('--devices', nargs='+', type=int, default=None,
                        help='GPU device IDs (e.g. 0 1). 미지정 시 auto')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_interval', type=int, default=5,
                        help='check_val_every_n_epoch (default: 5)')
    parser.add_argument('--debug',  action='store_true',
                        help='디버그 모드 (2 epoch, num_workers=0, single GPU)')
    parser.add_argument('--save_id', default=None,
                        help='실험 ID (default:[{sampling_rate}Hz_t{TIME_BIN}_s{STRIDE}_w{PATCH_SIZE}/])')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    IS_DEBUG = is_debugging()
    if IS_DEBUG:
        print("[Debug Mode] Single GPU")
        strategy = 'auto'
        devices = [0]
        num_workers = 0
    else:
        devices = args.devices if args.devices else 'auto'
        if isinstance(devices, list) and len(devices) == 1:
            strategy = 'auto'
            print(f"[Training Mode] Single GPU (device={devices[0]})")
        else:
            strategy = 'ddp'
            print(f"[Training Mode] DDP Strategy (devices={devices})")
        num_workers = args.num_workers

    STRIDE = args.stride
    TIME_BIN = args.time_bin
    PATCH_SIZE = args.patch_size

    config = {
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "accumulate_grad_batches": 1,
                "num_workers": num_workers,
                "shuffle": True,
                "sampling_rate": args.sampling_rate,
                "start_time_ms" : -200,
                "data_ext": "npy",
                "window_type": "fixed",  # "fixed" or "random"
                "time_bin": TIME_BIN,
                "file_chunk_type": "subject", # "subject" or "run"
                "normalize_method": "zscore", # "zscore" or "minmax"
                "patch_idx": None,
                "stride": STRIDE,
                "save_dir": None,
                "num_epochs": args.epochs if not IS_DEBUG else 2,
                "patience": args.patience,
                "n_classes": args.n_classes,
                "is_label_null": True,
                "skip_pred": False,
                "metrics": ["acc", "bal_acc"],
                "ckpt_dir": None,
                "test_files": None,
                "patch_size": PATCH_SIZE,
    }

    print(f"[ver2] batch_size={config['batch_size']}, patience={config['patience']}, "
          f"val_interval={args.val_interval}, label_smoothing=0.1, "
          f"optimizer=AdamW(wd=1e-2), scheduler=CosineAnnealingLR")

    # -------------------------------------------------------
    # Discover all subjects
    # -------------------------------------------------------
    subject_files = discover_all_subjects(config["data_dir"])
    subject_ids = sorted(subject_files.keys())

    if args.subject:
        subject_ids = [s for s in subject_ids if s in args.subject]

    print(f"[*] Found {len(subject_ids)} subjects: {subject_ids}")

    RESUME_MODE = not args.no_resume
    root_log_dir = f"./EEGNet/within_logs_ver2/{config['sampling_rate']}Hz_t{TIME_BIN}_s{STRIDE}_w{PATCH_SIZE}"
    train_date = datetime.now().strftime('%Y%m%d_%H%M')


    # -------------------------------------------------------
    # Subject Loop
    # -------------------------------------------------------
    for subject_id in subject_ids:

        # --- Resume: 완료된 subject 스킵 ---
        if RESUME_MODE:
            subj_log_dir = os.path.join(root_log_dir, subject_id)
            completed_patches = find_completed_patches(subj_log_dir)
            subj_files_tmp = subject_files[subject_id]
            input_len_tmp, _ = COMBDataset(
                config={**config, "test_files": subj_files_tmp},
                filepath=subj_files_tmp
            )._get_sample_info()
            n_patches_tmp = (input_len_tmp - TIME_BIN) // STRIDE + 1
            if len(completed_patches) >= n_patches_tmp:
                print(f"[SKIP] {subject_id}: all {n_patches_tmp} patches done")
                continue
            elif len(completed_patches) > 0:
                print(f"[RESUME] {subject_id}: {len(completed_patches)}/{n_patches_tmp} patches done, resuming...")

        print(f"\n{'='*60}")
        print(f"[*] Processing Subject: {subject_id}")
        print(f"{'='*60}")

        subj_files = subject_files[subject_id]
        config["test_files"] = subj_files

        input_len, input_ch = COMBDataset(
            config=config, filepath=subj_files
        )._get_sample_info()
        n_patches = (input_len - TIME_BIN) // STRIDE + 1

        config["n_patches"] = n_patches
        config["input_len"] = input_len
        config["input_ch"] = input_ch

        print(f"  input_len={input_len}, input_ch={input_ch}, "
              f"n_patches={n_patches} (stride={STRIDE})")

        # Trial-level Split
        train_trial_indices, val_trial_indices, test_trial_indices = \
            WithinSubjectDatasetTotal.split_trials(
                config=config, filepath=subj_files,
                val_size=0.2, test_size=0.2, random_state=42
            )

        print(f"  Train={len(train_trial_indices)}, "
              f"Val={len(val_trial_indices)}, "
              f"Test={len(test_trial_indices)}")

        # -------------------------------------------------------
        # Patch Loop
        # -------------------------------------------------------
        # Resume: 완료된 패치 목록 갱신
        if RESUME_MODE:
            subj_log_dir = os.path.join(root_log_dir, subject_id)
            completed_patches = find_completed_patches(subj_log_dir)
        else:
            completed_patches = set()

        for patch_idx in range(n_patches):
            if RESUME_MODE and patch_idx in completed_patches:
                continue
            target_cls = 'All'
            experiment_name = (f"Within_{subject_id}"
                               f"_P{patch_idx}"
                               f"_t{TIME_BIN}_s{STRIDE}")
            exp_id = f"{train_date}_{experiment_name}"

            base_dir = (f"./EEGNet/within_logs_ver2/"
                        f"{config['sampling_rate']}Hz_t{TIME_BIN}_s{STRIDE}_w{PATCH_SIZE}/"
                        f"{subject_id}/{exp_id}/")
            ckpt_dir = os.path.join(base_dir, "checkpoints")
            analysis_dir = os.path.join(base_dir, "analysis")

            config["save_dir"] = analysis_dir
            config["ckpt_dir"] = ckpt_dir
            config["patch_idx"] = patch_idx

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(analysis_dir, exist_ok=True)

            print(f"\n  [*] Patch {patch_idx}/{n_patches-1} | {experiment_name}")

            # Datasets
            print(f"  [DEBUG] Creating train_dataset...")
            train_dataset = WithinSubjectDatasetTotal(
                config=config, filepath=subj_files,
                trial_indices=train_trial_indices
            )
            print(f"  [DEBUG] train_dataset created, len={len(train_dataset)}")
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=config["shuffle"],
                num_workers=config["num_workers"],
                collate_fn=torch_collate_fn,
                persistent_workers=config["num_workers"] > 0,
                pin_memory=True,
            )
            print(f"  [DEBUG] train_loader created")
            print(f"  [DEBUG] Creating val_dataset...")
            val_dataset = WithinSubjectDatasetTotal(
                config=config, filepath=subj_files,
                trial_indices=val_trial_indices
            )
            print(f"  [DEBUG] val_dataset created, len={len(val_dataset)}")
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                collate_fn=torch_collate_fn,
                persistent_workers=config["num_workers"] > 0,
                pin_memory=True,
            )
            print(f"  [DEBUG] val_loader created")

            # Model
            torch.set_float32_matmul_precision('medium')

            # 모델 초기화 (LightningModule wrapper)
            model = LitEEGNet(
                n_channels=input_ch,
                n_timepoints=PATCH_SIZE,
                n_classes=config["n_classes"],
                config=config
            )
            print(f"FC input size: {model.model.fc_input_size}")

            # Logger
            wandb_logger = WandbLogger(
                project="eegnet_within_total_ver2",
                name=experiment_name,
                group=f"Within_{subject_id}",
                job_type="train",
                id=exp_id,
                tags=[f"Within_{subject_id}", f"Patch{patch_idx}",
                      f"Class{target_cls}", f"Stride{STRIDE}",
                      f"Timebin{TIME_BIN}", f"Window{PATCH_SIZE}", exp_id],
                save_dir=base_dir,
            )

            # Callbacks
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

            checkpoint_callback = ModelCheckpoint(
                monitor='valid_loss',
                mode='min',
                save_top_k=5,
                save_last=True,
                filename=f"{experiment_name}-{{epoch:02d}}-{{valid_loss:.4f}}",
                verbose=False,
                dirpath=ckpt_dir,
            )

            early_stop_callback = EarlyStopping(
                monitor='valid_loss',
                min_delta=0.00,
                patience=args.patience,
                verbose=True,
                mode='min',
            )

            callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

            if is_debugging():
                logger = []
            else:
                wandb_logger.watch(model, log_graph=True)
                logger = [wandb_logger]

            # Trainer
            trainer = pl.Trainer(
                accelerator='cuda',
                devices=devices,
                strategy=strategy,
                sync_batchnorm=False,
                precision="16-mixed",
                max_epochs=config["num_epochs"],
                accumulate_grad_batches=config["accumulate_grad_batches"],
                callbacks=callbacks,
                enable_progress_bar=True,
                num_sanity_val_steps=0,
                check_val_every_n_epoch=1 if IS_DEBUG else args.val_interval,
                logger=logger,
            )

            trainer.fit(model, train_loader, val_loader)
            print(f"  [DEBUG] trainer.fit completed!")

            wandb.finish()
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        print(f"\n[*] Subject {subject_id} finished.")

    print(f"\n[!] All Subjects Training Completed.")
