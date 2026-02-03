
import sys
import os
import torch
from datetime import datetime
from EEGPT.finetune_EEGPT_combine_LoRA_conv_within_util import LitEEGPTCausal_LoRA, WithinSubjectDataset
from utils_my import InferenceManager, COMBDataset, torch_collate_fn, get_patch_time_ms
import torch.nn.functional as F
import numpy as np
import re

class EEGPTInference(InferenceManager):
    def __init__(self, config, model, test_trial_indices=None):
        self.test_trial_indices = test_trial_indices
        super().__init__(config, model)
    def _discover_checkpoints(self):

        root_ckpt_dir = self.config["ckpt_dir"]

        if not os.path.exists(root_ckpt_dir):
            print(f" Checkpoint dir not found: {root_ckpt_dir}")
            return []
        
        infer_date = 20260131
        best_model_paths = []

        # 2. 조건에 맞는 모델 폴더 필터링 (날짜 >= infer_date, F1 포함)
        all_dirs = os.listdir(root_ckpt_dir)
        model_ckpt_list = []
        for m in all_dirs:
            # parts = m.split("_")
            # # 폴더명이 날짜로 시작하고, 길이가 충분하며, 조건을 만족하는지 확인
            # if (parts[0].isdigit() 
            #     and int(parts[0]) >= infer_date 
            #     and len(parts) > 3 
            #     and parts[3] == "F1"):
            #     model_ckpt_list.append(m)
            if m == '20260203_1429_LORA_F1_P5_CAll':
                model_ckpt_list.append(m)
        
        model_ckpt_list.sort()

        best_model_paths = []

        for model_ckpt in model_ckpt_list:
 
            ckpt_path = os.path.join(root_ckpt_dir, model_ckpt, "checkpoints")

            if not os.path.exists(ckpt_path):
                print(f"  [Skip] No checkpoints dir found: {ckpt_path}")
                continue

            # 예: patch_0, patch_1 폴더 내부의 .pth 파일등 중 best만 찾거나 전체 리스트 업
            # 사용자 파일 구조에 맞춰 패턴 사용

            ckpt_lists = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")]
            ckpt_lists.sort()
            
            min_loss = float('inf')  # Loss는 낮을수록 좋으므로 무한대로 초기화
            best_file = None

            # patch_dir = os.path.join(self.config['ckpt_dir'], c)
            for c in ckpt_lists:
                try:
                    # 파일명 예시: ...loss=1.7780.ckpt 라고 가정
                    if "loss=" in c:
                        # loss 뒤의 숫자를 파싱 (.ckpt 제거)
                        loss_str = c.split("loss=")[-1].replace(".ckpt", "")
                        c_loss = float(loss_str)

                        if c_loss < min_loss:
                            min_loss = c_loss
                            best_file = c
                except Exception as e:
                    print(f"  [Warning] Parsing failed for {c}: {e}")
                    continue
            if best_file:
                # 중요: root_ckpt_dir가 아니라 현재 순회 중인 ckpt_path와 합쳐야 함
                full_path = os.path.join(ckpt_path, best_file)
                best_model_paths.append(full_path)
                # print(f"  [Found] Best Loss: {min_loss:.4f} -> {best_file}")
            else:
                print(f"  [Skip] No valid checkpoint file found in {ckpt_path}")

        # best는 여기서 구현
        # epoch으로 고정하던지 best만 선택하는 함수를 추가 구현하기
        return best_model_paths


    def _parse_patch_from_path(self, filepath):
        match = re.search(r"_P(\d+)_", filepath)
        if match:
            return int(match.group(1))
        print(f"[Warning] Failed to extract patch number from: {filepath}")
        return 0
    
    def _load_model_instance(self, ckpt_path, net):
        # 1. Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu') # GPU tensors are moved to CPU

        # 2. Extract the model's state_dict
        # PyTorch Lightning saves the 'state_dict' under a different key.
        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint: # Legacy PyTorch save
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint # Fallback to the checkpoint itself

        # 3. Load the model
        # strict=False: LoRA layers are not loaded into the model (only the base model)
        msg = net.load_state_dict(state, strict=False)
        print(f"[*] Model Loaded: {msg}") # missing_keys and unexpected_keys are printed

        net.cuda().eval()
        return net
    
# [1] 디버거가 붙어있는지 확인하는 함수
def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None

if __name__ == "__main__":

    # [2] 상황에 따른 전략 설정
    if is_debugging():
        print("🚀 [Debug Mode Detected] Switching to Single GPU (strategy='auto')")
        strategy = 'auto'
        devices = [0]
        num_workers = 1
    else:
        print("⚡ [Training Mode] Using DDP Strategy")
        strategy = 'ddp'
        devices = 'auto'
        num_workers = 16
    IS_DEBUG = is_debugging()

    test_config = {
                "data_dir": "./EEG(256Hz)_COMB/processed_test/npy",
                "batch_size": 64,
                "num_workers": 16,
                "shuffle": True,
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

    test_files = [
            os.path.join(test_config["data_dir"], f) for f in os.listdir(test_config["data_dir"])
            if "label" not in f and "stats" not in f and "info" not in f
        ]

    test_config["test_files"] = test_files
    input_len, input_ch = COMBDataset(config=test_config, filepath=test_config["test_files"])._get_sample_info()
    n_patches = input_len // test_config["time_bin"]

    test_config["n_patches"] = n_patches
    test_config["input_len"] = input_len
    test_config["input_ch"] = input_ch


    # Init model (LightningModule) — patch_idx=0 as placeholder,
    # InferenceManager.run() iterates all patches internally
    model = LitEEGPTCausal_LoRA(
        config=test_config,
        fixed_train_patch_idx=0,
    )

    manager = EEGPTInference(test_config, model=model)
    manager.run()

    del model, manager
    torch.cuda.empty_cache()
