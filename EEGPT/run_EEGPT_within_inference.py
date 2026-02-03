
import sys
import os
import torch
from datetime import datetime
from EEGPT.finetune_EEGPT_combine_LoRA_conv_within_util import LitEEGPTCausal_LoRA, WithinSubjectDataset
from utils_my import InferenceManager, COMBDataset, torch_collate_fn, get_patch_time_ms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re


class EEGPTWithinInference(InferenceManager):
    def __init__(self, config, model, test_trial_indices=None):
        self.test_trial_indices = test_trial_indices
        super().__init__(config, model)

    def _discover_checkpoints(self):

        root_ckpt_dir = self.config["ckpt_dir"]

        if not os.path.exists(root_ckpt_dir):
            print(f" Checkpoint dir not found: {root_ckpt_dir}")
            return []

        # Within-Subject 모델만 필터링
        all_dirs = os.listdir(root_ckpt_dir)
        model_ckpt_list = []
        for m in all_dirs:
            if 'Within' in m:
                model_ckpt_list.append(m)

        model_ckpt_list.sort()

        best_model_paths = []

        for model_ckpt in model_ckpt_list:

            ckpt_path = os.path.join(root_ckpt_dir, model_ckpt, "checkpoints")

            if not os.path.exists(ckpt_path):
                print(f"  [Skip] No checkpoints dir found: {ckpt_path}")
                continue

            ckpt_lists = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")]
            ckpt_lists.sort()

            min_loss = float('inf')
            best_file = None

            for c in ckpt_lists:
                try:
                    if "loss=" in c:
                        loss_str = c.split("loss=")[-1].replace(".ckpt", "")
                        c_loss = float(loss_str)

                        if c_loss < min_loss:
                            min_loss = c_loss
                            best_file = c
                except Exception as e:
                    print(f"  [Warning] Parsing failed for {c}: {e}")
                    continue
            if best_file:
                full_path = os.path.join(ckpt_path, best_file)
                best_model_paths.append(full_path)
            else:
                print(f"  [Skip] No valid checkpoint file found in {ckpt_path}")

        return best_model_paths


    def _parse_patch_from_path(self, filepath):
        match = re.search(r"_P(\d+)_", filepath)
        if match:
            return int(match.group(1))
        print(f"[Warning] Failed to extract patch number from: {filepath}")
        return 0

    def _load_model_instance(self, ckpt_path, net):
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint

        msg = net.load_state_dict(state, strict=False)
        print(f"[*] Model Loaded: {msg}")

        net.cuda().eval()
        return net

    def predict_within_subject(self, model, test_patch_idx):
        """
        Within-Subject용 predict: test_trial_indices를 사용하여 테스트
        """
        test_config = self.config.copy()
        test_config["patch_idx"] = test_patch_idx

        # WithinSubjectDataset 사용 (test_trial_indices로 필터링)
        test_dataset = WithinSubjectDataset(
            config=test_config,
            filepath=self.config["test_files"],
            trial_indices=self.test_trial_indices
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=torch_collate_fn
        )

        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, _, mask in test_loader:
                inputs = inputs.cuda(0)
                outputs = model(inputs, mask=mask)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    logits = outputs[-1]
                else:
                    logits = outputs
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def run(self):
        """Within-Subject용 run 메서드"""
        print(f" Start Within-Subject Inference")
        print(f"   - Found {len(self.ckpt_paths)} checkpoints")
        print(f"   - Test Trials: {len(self.test_trial_indices) if self.test_trial_indices else 'All'}")
        print(f"   - Total Test Patches: {self.num_patches}")

        global_tgm_data = []

        for ckpt in self.ckpt_paths:
            print(f"\n==> Evaluating Checkpoint: {ckpt}")
            train_patch_idx = self._parse_patch_from_path(ckpt)
            train_start_ms, train_end_ms = get_patch_time_ms(train_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
            print(f"Train Patch{train_patch_idx}({train_start_ms}~{train_end_ms})")

            save_dir = os.path.dirname(ckpt).replace("checkpoints", "analysis")
            model = self.model
            if not self.config["skip_pred"]:
                model = self._load_model_instance(ckpt, self.model)

            for test_patch_idx in range(self.num_patches):
                test_start_ms, test_end_ms = get_patch_time_ms(test_patch_idx, self.config['time_bin'], self.config['sampling_rate'])
                print(f"Test Patch{test_patch_idx}({test_start_ms}~{test_end_ms})")

                if not self.config["skip_pred"]:
                    preds, labels, probs = self.predict_within_subject(model, test_patch_idx)
                    self.recorder.save_detail_csv(save_dir, train_patch_idx, test_patch_idx, preds, labels, probs)
                else:
                    preds, labels, probs = self.loader.load_csv(train_patch_idx, test_patch_idx)

                metrics = self.evaluator.compute_metrics(preds, labels, probs, params=self.config["metrics"])
                self.visualizer.plot_cm(save_dir, labels, preds, train_patch_idx, test_patch_idx, metrics['acc'])

                global_tgm_data.append({
                    'train_patch_idx': train_patch_idx,
                    'train_time_ms': train_start_ms,
                    'test_patch_idx': test_patch_idx,
                    'test_time_ms': test_start_ms,
                    'test_acc': metrics['acc'],
                    'test_bal_acc': metrics['bal_acc']
                })

        if model: del model
        torch.cuda.empty_cache()

        if global_tgm_data:
            df_summary = pd.DataFrame(global_tgm_data)
            self.recorder.save_summary_csv(df_summary, save_dir=save_dir)
            self.visualizer.plot_tgm(ckpt, df_summary, metric_key='test_acc', save_dir=save_dir)
            self.visualizer.plot_tgm(ckpt, df_summary, metric_key='test_bal_acc', save_dir=save_dir)

        return


# [1] 디버거가 붙어있는지 확인하는 함수
def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None

if __name__ == "__main__":

    # [2] 상황에 따른 전략 설정
    if is_debugging():
        print("[Debug Mode Detected] Switching to Single GPU")
        num_workers = 1
    else:
        print("[Inference Mode]")
        num_workers = 16

    test_config = {
                "data_dir": "./EEG(256Hz)_COMB/",
                "batch_size": 64,
                "num_workers": num_workers,
                "shuffle": False,
                "sampling_rate": 256,
                "start_time_ms" : -200,
                "data_ext": "npy",
                "window_type": "fixed",
                "time_bin": 16,
                "file_chunk_type": "subject",
                "normalize_method": "zscore",
                "patch_idx": None,
                "stride": None,
                "save_dir": "EEGPT/logs",
                "num_epochs": 100,
                "patience": 10,
                "n_classes": 6,
                "is_label_null": True,
                "skip_pred" : False,
                "metrics":["acc", "bal_acc"],
                "ckpt_dir":"./EEGPT/logs",
                "csv_input_dir":None,
    }

    # Within-Subject: sub-46 파일만 사용
    train_dir = os.path.join(test_config["data_dir"], "processed_train/npy")
    test_files = [
            os.path.join(train_dir, f) for f in os.listdir(train_dir)
            if "label" not in f and "stats" not in f and "info" not in f and "sub-46" in f
        ]

    test_config["test_files"] = test_files

    # Trial Split (학습 때와 동일한 random_state 사용!)
    train_trial_indices, val_trial_indices, test_trial_indices = WithinSubjectDataset.split_trials(
        config=test_config,
        filepath=test_files,
        val_size=0.2,
        test_size=0.2,
        random_state=42  # 학습 때와 동일!
    )

    print(f"\n[*] Using Test Trial Indices: {len(test_trial_indices)} trials")

    input_len, input_ch = COMBDataset(config=test_config, filepath=test_files)._get_sample_info()
    n_patches = input_len // test_config["time_bin"]

    test_config["n_patches"] = n_patches
    test_config["input_len"] = input_len
    test_config["input_ch"] = input_ch

    # Init model
    model = LitEEGPTCausal_LoRA(
        config=test_config,
        fixed_train_patch_idx=0,
    )

    manager = EEGPTWithinInference(test_config, model=model, test_trial_indices=test_trial_indices)
    manager.run()

    del model, manager
    torch.cuda.empty_cache()
