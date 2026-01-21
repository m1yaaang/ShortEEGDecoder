import random 
import os
from types import NoneType
import pickle

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import numpy as np
import pandas as pd

import tqdm
from functools import partial
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')

from utils import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from peft import LoraConfig, get_peft_model

from finetune_EEGPT_combine_LoRA import (
                        COMBLoader,
                        LitEEGPTCausal,
)       

##SECTION - MAIN
# %%MAIN
if __name__ == "__main__":
    # load configs

    # Train Data Num : 6Class: 1 2 3 4 5 6
    # 1 is null, 정확도에는 포함 안함

    # subjects = [22]
    # N = len(subjects)//10
    # set_all = set(subjects)

    # target_cls = range(1,5)
    # train_patch = range()
    IS_DEBUG = False

    #FIXME - DEBUG
    # IS_DEBUG = True
    # target_cls = 2
    train_patch_idx = [5,6,7,8]
    # train_patch_idx = [7]

    ##FIXME - CONFIG(experiment setting)
    # ------------------config---------------------------
    time_bin = 16   # 16 timepoints = 62.5ms/ 64 timepoints = 250ms
    # batch_size = 8*4
    batch_size = 4  # 메모리 부족 문제로 줄임 (기존 32)
    accumulate_grad_batches = 8  # batch_size 줄인만큼 늘림 (기존 4)
    max_epochs = 50 if not IS_DEBUG else 2 # 디버그 시 에포크 단축

    # Classes: Original labels are 1-6, after (label-1) in dataset: 0-5                                                   
    # Class 0 (original 1) = Null -> always "Rest"           
    # Classes 1-5 (original 2-6) = Targets for OVR           
    TARGET_CLASSES = [1, 2, 3, 4, 5]  # After label-1 transformation   
    TOTAL_PATCHES=[256//time_bin]
    k_folds=2
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # ------------------config---------------------------



    root_dir = "./PreprocessedEEG/"
    train_files = os.listdir(os.path.join(root_dir, "processed_train"))
    test_files = os.listdir(os.path.join(root_dir, "processed_test"))

    #train은 fold결과에 따라 달라짐
    test_dataset = COMBLoader(
        os.path.join(
            root_dir, "processed_test"), test_files, is_train=False, time_bin=time_bin
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    train_date = datetime.now().strftime('%Y%m%d_%H%M')

    # 폴드 결과 저장
    fold_results = []


    ckpt_list = [ '/local_raid3/03_user/myyu/EEGPT/downstream_combine3/logs/20260114_2210_LoRa_Timebin16_F1_P5_CAll/checkpoints/LoRa_MultiC_Timebin16-epoch=40-valid_loss=1.7888.ckpt',
                    '/local_raid3/03_user/myyu/EEGPT/downstream_combine3/logs/20260114_2210_LoRa_Timebin16_F1_P6_CAll/checkpoints/LoRa_MultiC_Timebin16-epoch=45-valid_loss=1.7751.ckpt',
                    '/local_raid3/03_user/myyu/EEGPT/downstream_combine3/logs/20260114_2210_LoRa_Timebin16_F1_P7_CAll/checkpoints/LoRa_MultiC_Timebin16-epoch=49-valid_loss=1.7739.ckpt',
                    '/local_raid3/03_user/myyu/EEGPT/downstream_combine3/logs/20260114_2210_LoRa_Timebin16_F1_P8_CAll/checkpoints/LoRa_MultiC_Timebin16-epoch=42-valid_loss=1.7844.ckpt'
    ]

    target_fold = 0
    # -------------------------------------------------------
    # Loop 1: K fold
    # -------------------------------------------------------
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_files)):
    # for fold in range(1)
        current_fold = fold + 1
        
        # -------------------------------------------------------
        # Loop 2: Patches (For Each Time Bin)
        # -------------------------------------------------------
        #FIXME - DEBUG(patch)
        # for train_patch_idx in TOTAL_PATCHES:
        for i, patch_idx in enumerate(train_patch_idx):

            # -------------------------------------------------------
            # Loop 3: Target Class (OVR Binary Classifiers)
            # -------------------------------------------------------    
            #FIXME - DEBUG(Target Class)        
            # for target_cls in TARGET_CLASSES:
            # for target_cls in [target_cls]:
                target_cls = 'All'
                ##FIXME - CONFIG(project name)
                # ------------------config---------------------------

                exp_id = ckpt_list[i].split('/logs/')[1].split('/')[0]
                experiment_name = f'continue_{exp_id}'
                # experiment_name = f"LORA_Timebin{time_bin}_F{current_fold}_P{patch_idx}_C{target_cls}"
                # exp_id = f"{train_date}_{experiment_name}"
                print(f"[*] Experiment ID: {exp_id}")
                print(f"[*] Mode: {'DEBUG (Overfit 1 batch)' if IS_DEBUG else 'TRAINING'}")

                base_dir = f"./logs/{exp_id}/continue/"
                ckpt_dir = os.path.join(base_dir,"checkpoints")
                analysis_dir = os.path.join(base_dir,"analysis")

                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(analysis_dir, exist_ok=True)

                print(f"  Training Class:{target_cls}")
                
                
                # 2. Fold에 맞춰 파일 리스트 나누기
                train_files_fold = [train_files[i] for i in train_ids]
                val_files_fold = [train_files[i] for i in val_ids]

                train_dataset = COMBLoader(os.path.join(root_dir, "processed_train"), 
                                            train_files_fold, is_train=True, time_bin=time_bin)
                val_dataset = COMBLoader(os.path.join(root_dir, "processed_train"),
                                        val_files_fold, is_train=False, time_bin=time_bin)
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,persistent_workers=True)


                #ANCHOR - Training
                # -- begin Training ------------------------------

                import math
                torch.set_float32_matmul_precision('medium' )



                # 4-3. init model(MultiClass)
                model = LitEEGPTCausal(
                    fixed_train_patch_idx=patch_idx,
                    load_path = None
                )
                checkpoint = torch.load(ckpt_list[i])
                model.load_state_dict(checkpoint['state_dict'],strict=False)
                # msg = model.load_state_dict(checkpoint['state_dict'], strict=False)

                # print("Missing keys:", msg.missing_keys)
                # print(model)
                wandb_logger = WandbLogger(
                                    project="eegpt_combine3_LoRa", 
                                    name=experiment_name,
                                    group="contiue",
                                    tags=[exp_id,experiment_name,f"Patch{patch_idx}",f"Class{target_cls}","Masking"],
                                    save_dir=base_dir
                                    )

                # wandb_logger.experiment.config.update({"max_epochs": max_epochs})
                # -------------------config--------------------------

                # 4-4. Loggers
                wandb_logger.watch(model, log="all", log_graph=True, log_freq=100)
                # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
                lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

                checkpoint_filename = f"{experiment_name}_Fold{current_fold}-{{epoch:02d}}-{{valid_loss:.4f}}"

                checkpoint_callback = ModelCheckpoint(
                    monitor='valid_loss',
                    mode='min',
                    save_top_k=5,
                    save_last=True,
                    filename=f"{experiment_name}-{{epoch:02d}}-{{valid_loss:.4f}}",  
                    verbose=False,
                    dirpath=ckpt_dir,
                )
                            
                callbacks = [lr_monitor, checkpoint_callback]

            
                ''' 
                !) pl.Trainer is for LightningModule 
                
                1. manage hardware => accelerator
                2. optimization => Mixed Precision
                3. Manage Train loop => max_epoch
                4. Logging => logger
                5. checkpoint, learning Rate monitor -> callback
                '''
                #FIXME - Trainer
                debug = None
                trainer = pl.Trainer(accelerator='cuda',
                                    devices=[0,1],
                                    precision='16-mixed',
                                    max_epochs=max_epochs, 
                                    accumulate_grad_batches = accumulate_grad_batches,
                                    callbacks=callbacks,
                                    enable_progress_bar=True,
                                    num_sanity_val_steps=0,
                                    check_val_every_n_epoch=10,
                                    logger=[wandb_logger, 
                                            pl_loggers.CSVLogger(base_dir, name="EEGPT_COMBINE_csv")])

                ##For Debug Dataset and network
                # IS_DEBUG = True
                # trainer=pl.Trainer(accelerator='gpu',
                #                     devices=[0],
                #                     precision=16,
                #                     max_epochs=1,
                #                     accumulate_grad_batches = accumulate_grad_batches,
                #                     limit_train_batches=1,
                #                     limit_val_batches=0,
                #                     num_sanity_val_steps=0,
                #                     logger=False,
                #                     enable_checkpointing=False)

                # For Debug Overfit one Batch
                # IS_DEBUG = True
                # trainer = pl.Trainer(
                #     accelerator='gpu', devices=[0],
                #     max_epochs=max_epochs,
                #     accumulate_grad_batches = accumulate_grad_batches,
                #     precision=16,
                #     num_sanity_val_steps=0,
                #     overfit_batches=1,
                #     logger=wandb_logger,
                #     enable_progress_bar=False
                # )

                trainer.fit(model, train_loader, val_loader)

                wandb.finish()
                
                if not IS_DEBUG: # INFERENCE
                    best_score = trainer.checkpoint_callback.best_model_score
                    print(f"OVR_Class{target_cls}_Patch{patch_idx}_Timebin{time_bin} Fold {fold+1} Best Validation Loss:{best_score}")
                    fold_results.append(best_score.item())

                    #FIXME - 이 모델로 전체 시간대 테스트 수행 후 저장
                    # analyze_temporal_importance(model, test_loader, ...)

                    # ckpt_dir_template = f"./log/{exp_id}_Fold{fold+1}/OVR_Class{{}}_Patch{{}}_Timebin{time_bin}"
                    # analyze_TGM()
                    trainer.save_checkpoint(f"{ckpt_dir}/{checkpoint_filename}.ckpt")
                    
                    wandb_logger.experiment.finish()
                    del model, trainer, train_loader, val_loader
                    torch.cuda.empty_cache()
                    

                # analyze_temporal_importance(model, valid_loader, save_dir=analysis_dir, mode="valid")
                # analyze_temporal_importance(model, test_loader, save_dir=analysis_dir)

        # -------------------------------------------------------
        # END Loop 2: Patches (For Each Time Bin)
        # -------------------------------------------------------
        break
    if not IS_DEBUG and fold_results:
        print(f"\n[!] {k_folds}-Fold CV Finished.")
        print(f"Average Val Loss: {np.mean(fold_results):.4f}")

#!SECTION
#endregion [!] MAIN