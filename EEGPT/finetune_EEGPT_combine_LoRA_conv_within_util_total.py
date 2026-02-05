'python EEGPT/finetune_EEGPT_combine_LoRA_conv_within_util_total.py'

import random
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split

from EEGPT.finetune_EEGPT_combine_LoRA_conv_within_util import LitEEGPTCausal_LoRA
from utils_my import COMBDataset, torch_collate_fn, NpyFileHandler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')


# -------------------------------------------------------
# Stride-aware time conversion
# -------------------------------------------------------
def get_patch_time_ms_stride(patch_idx, time_bin, stride, sampling_rate, start_time_ms=-200):
    """
    stride를 고려한 패치 인덱스 -> 실제 시간(ms) 변환.
    기존 get_patch_time_ms는 stride=time_bin을 가정하지만,
    이 함수는 stride가 time_bin과 다를 때도 올바르게 계산.
    """
    stride_ms = stride / sampling_rate * 1000
    window_ms = time_bin / sampling_rate * 1000
    start_ms = patch_idx * stride_ms + start_time_ms
    end_ms = start_ms + window_ms
    return start_ms, end_ms


# -------------------------------------------------------
# WithinSubjectDatasetTotal: global index fix
# -------------------------------------------------------
class WithinSubjectDatasetTotal(COMBDataset):
    """
    COMBDataset을 상속받아 trial 단위로 train/test split을 지원하는 Dataset.
    Multi-file subject에서도 global trial index를 올바르게 처리.
    데이터를 메모리에 캐싱하여 I/O 병목 제거.
    """
    def __init__(self, config, filepath=None, trial_indices=None, shared_cache=None):
        self.selected_trial_indices = trial_indices
        self._cache = shared_cache if shared_cache is not None else {}
        super().__init__(config, filepath)
        self._preload_cache()

    def _preload_cache(self):
        """모든 파일의 mmap 참조를 캐싱 (매 __getitem__마다 np.load 호출 방지)"""
        for path in self.file_paths:
            if path not in self._cache:
                y_path = path.replace('.npy', '_label.npy')
                stat_path = path.replace('.npy', '_stats.npy')
                self._cache[path] = (
                    np.load(path, mmap_mode='r'),
                    np.load(y_path, mmap_mode='r'),
                    np.load(stat_path, mmap_mode='r'),
                )

    def __getitem__(self, index):
        """캐싱된 mmap에서 직접 읽어 I/O 오버헤드 제거"""
        file_idx, trial_idx = self.trial_map[index]
        path = self.file_paths[file_idx]

        start, end = self.window_type(total_len=self.model_input_len)

        X_all, Y_all, stat_all = self._cache[path]
        X = X_all[trial_idx, :, start:end]
        Y = Y_all[trial_idx]
        stat = stat_all[trial_idx]

        X = X[np.newaxis, :, :]
        stat = stat[np.newaxis, :, :]

        mask, X = self._masking_from_window(self.model_input_len, X, start, end)
        X = self.normalizer(X, stat, mask)

        X = torch.from_numpy(X).float()
        Y = torch.tensor(Y).long().unsqueeze(0)
        stat = torch.from_numpy(stat.copy()).float()
        mask = torch.from_numpy(mask).bool()

        return X, Y, stat, mask

    def _build_trial_map(self):
        """
        Global index 기반 trial_map 빌드.
        split_trials가 생성한 global indices를 (file_idx, local_trial_idx)로 변환.
        """
        global_offset = 0
        for f_idx, path in enumerate(self.file_paths):
            try:
                shape = self.file_handler.get_meta(path)
                n_trials = shape[0]

                if self.selected_trial_indices is not None:
                    for global_t_idx in self.selected_trial_indices:
                        local_idx = global_t_idx - global_offset
                        if 0 <= local_idx < n_trials:
                            self.trial_map.append((f_idx, local_idx))
                else:
                    for t_idx in range(n_trials):
                        self.trial_map.append((f_idx, t_idx))

                global_offset += n_trials
            except Exception as e:
                print(f"[Warning] Error reading {path}: {e}")

    @staticmethod
    def split_trials(config, filepath, val_size=0.2, test_size=0.2, random_state=42):
        """trial indices를 train/val/test로 분할"""
        file_handler = NpyFileHandler()
        total_trials = 0
        for path in filepath:
            shape = file_handler.get_meta(path)
            total_trials += shape[0]

        all_indices = list(range(total_trials))

        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=test_size,
            random_state=random_state, shuffle=True
        )

        val_ratio_in_trainval = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_in_trainval,
            random_state=random_state, shuffle=True
        )

        print(f"[*] Trial Split: Total={total_trials}, "
              f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        return train_indices, val_indices, test_indices


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(7)


class LitEEGPTCausal_LoRA_Total(LitEEGPTCausal_LoRA):
    """val CM을 파일 저장 없이 wandb에만 로깅하는 서브클래스"""

    def on_validation_epoch_end(self):
        import torch
        from pytorch_lightning.loggers import WandbLogger

        if not self.running_scores["valid"]:
            return

        label_list = [l.to(self.device) for l, s in self.running_scores["valid"]]
        score_list = [s.to(self.device) for l, s in self.running_scores["valid"]]

        labels = torch.cat(label_list, dim=0)
        logits = torch.cat(score_list, dim=0)

        labels = self.all_gather(labels)
        logits = self.all_gather(logits)

        labels = labels.cpu().numpy()
        logits = logits.float()

        best_threshold = 0.5

        if self.target_class_idx is not None:
            probs = torch.sigmoid(logits).cpu()
            labels = labels.astype(int)
            preds = (probs >= best_threshold).long().numpy()
            probs = probs.numpy()
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)

        metric_results = self.evaluator.compute_metrics(
            preds=preds, labels=labels, probs=probs,
            params=self.metrics_list
        )
        for metric_name, value in metric_results.items():
            self.log(f"valid_{metric_name}", value, on_epoch=True, prog_bar=True)

        # Confusion Matrix: wandb only (no file save)
        should_log_image = (self.current_epoch + 1) % 10 == 0
        if should_log_image and isinstance(self.logger, WandbLogger) and self.global_rank == 0:
            label_names = np.unique(labels)
            cm = confusion_matrix(labels, preds, labels=label_names)
            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=[f"Class_{i}" for i in label_names],
                        yticklabels=[f"Class_{i}" for i in label_names])
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            ax_cm.set_title(f"Valid CM - Epoch {self.current_epoch+1} "
                            f"P{self.hparams.config['patch_idx']} "
                            f"| Acc: {metric_results['acc']:.2f}%")
            ax_cm.invert_yaxis()
            plt.tight_layout()
            self.logger.experiment.log({
                "valid/confusion_matrix_img": wandb.Image(fig_cm),
                "global_step": self.global_step
            })
            plt.close('all')

        self.running_scores["valid"] = []


def discover_all_subjects(data_dir):
    """
    processed_train/npy와 processed_test/npy에서
    모든 피험자 파일을 찾아 {subject_id: [file_paths]} 형태로 반환.
    """
    train_dir = os.path.join(data_dir, "processed_train/npy")
    test_dir = os.path.join(data_dir, "processed_test/npy")

    subject_files = {}

    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if (f.endswith('.npy')
                and 'label' not in f
                and 'stats' not in f
                and 'info' not in f):
                subject_id = f.replace('.npy', '')
                filepath = os.path.join(d, f)
                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                subject_files[subject_id].append(filepath)

    return subject_files


def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":

    if is_debugging():
        print("[Debug Mode] Single GPU")
        strategy = 'auto'
        devices = [0]
        num_workers = 1
    else:
        print("[Training Mode] DDP Strategy")
        strategy = 'ddp_find_unused_parameters_true'
        devices = 'auto'
        num_workers = 8

    IS_DEBUG = is_debugging()

    STRIDE = 4
    TIME_BIN = 16

    config = {
        "data_dir": "./EEG(500Hz)_53ch/",
        "batch_size": 2048,
        "accumulate_grad_batches": 1,
        "num_workers": num_workers,
        "shuffle": True,
        "sampling_rate": 500,
        "start_time_ms": -200,
        "data_ext": "npy",
        "window_type": "fixed",
        "time_bin": TIME_BIN,
        "file_chunk_type": "subject",
        "normalize_method": "zscore",
        "patch_idx": None,
        "stride": STRIDE,
        "num_epochs": 100 if not IS_DEBUG else 2,
        "patience": 10,
        "n_classes": 6,
        "is_label_null": True,
        "skip_pred": False,
        "metrics": ["acc", "bal_acc"],
        "save_dir": None,
        "ckpt_dir": None,
        "csv_input_dir": None,
        "test_files": None,
    }

    # -------------------------------------------------------
    # Discover all subjects
    # -------------------------------------------------------
    subject_files = discover_all_subjects(config["data_dir"])
    subject_ids = sorted(subject_files.keys())
    print(f"[*] Found {len(subject_ids)} subjects: {subject_ids}")

    train_date = datetime.now().strftime('%Y%m%d_%H%M')

    # -------------------------------------------------------
    # Subject Loop
    # -------------------------------------------------------
    for subject_id in subject_ids:
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
        for patch_idx in range(n_patches):
            target_cls = 'All'
            experiment_name = (f"LORA_Within_{subject_id}"
                               f"_P{patch_idx}"
                               f"_t{TIME_BIN}_s{STRIDE}")
            exp_id = f"{train_date}_{experiment_name}"

            base_dir = (f"./EEGPT/within_total/"
                        f"{config['sampling_rate']}Hz_t{TIME_BIN}_s{STRIDE}/"
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
            train_dataset = WithinSubjectDatasetTotal(
                config=config, filepath=subj_files,
                trial_indices=train_trial_indices
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=config["shuffle"],
                num_workers=config["num_workers"],
                collate_fn=torch_collate_fn,
                persistent_workers=True,
                pin_memory=True,
            )

            val_dataset = WithinSubjectDatasetTotal(
                config=config, filepath=subj_files,
                trial_indices=val_trial_indices
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                collate_fn=torch_collate_fn,
                persistent_workers=True,
                pin_memory=True,
            )

            # Model
            torch.set_float32_matmul_precision('medium')

            model = LitEEGPTCausal_LoRA_Total(
                config=config,
                fixed_train_patch_idx=patch_idx,
            )

            # Logger
            wandb_logger = WandbLogger(
                project="eegpt_combine_LoRa_within_total",
                name=experiment_name,
                group=f"Within_{subject_id}",
                job_type="train",
                id=exp_id,
                tags=[f"Within_{subject_id}", f"Patch{patch_idx}",
                      f"Class{target_cls}", f"Stride{STRIDE}",
                      f"Timebin{TIME_BIN}", exp_id],
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
                patience=5,
                verbose=True,
                mode='min',
            )

            callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

            if is_debugging():
                logger = [pl_loggers.CSVLogger(base_dir, name="csv_log")]
            else:
                wandb_logger.watch(model, log="all", log_graph=True, log_freq=100)
                logger = [wandb_logger,
                          pl_loggers.CSVLogger(base_dir, name="csv_log")]

            # Trainer
            trainer = pl.Trainer(
                accelerator='cuda',
                devices=devices,
                strategy=strategy,
                sync_batchnorm=True,
                precision=16,
                max_epochs=config["num_epochs"],
                accumulate_grad_batches=config["accumulate_grad_batches"],
                callbacks=callbacks,
                enable_progress_bar=True,
                num_sanity_val_steps=0,
                check_val_every_n_epoch=1 if IS_DEBUG else 10,
                logger=logger,
            )

            trainer.fit(model, train_loader, val_loader)

            wandb.finish()
            wandb_logger.experiment.finish()
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        print(f"\n[*] Subject {subject_id} finished.")

    print(f"\n[!] All Subjects Training Completed.")
