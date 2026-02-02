'torchrun --nproc_per_node=2 finetune_EEGPT_combine_LoRA.py'

import random 
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
from types import NoneType
import pickle
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
from utils import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from peft import LoraConfig, get_peft_model
from utils_my import COMBDataset, InferenceManager, torch_collate_fn, Evaluator, Visualizer

matplotlib.use('Agg')
# INSTALL `pip install peft`

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

# original code
# use_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ' ]
#


'''
CHANNEL_DICT = {k.upper():v for v,k in enumerate(       # 62 ch
                     [      'FP1', 'FPZ', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                               'O1', 'OZ', 'O2', ])}

## /home/winter/eegdecoder/minyung/EEGPT/downstream_combine/utils.py
# '''
# class ChannelDropDataset(COMBDataset):
#     def __init__(self, config, filepath, drop_channels=None, **kwargs):
#         """
#         Args:
#             config: 기존 config
#             drop_channels (list): 제외할 채널의 리스트 (예: [Fp1, Fp2])
#             **kwargs: filepath 등 부모 클래스 인자
#         """
#         self.drop_channels = drop_channels
#         self.drop_indices = []
#         self.keep_indices = []

        
#         # 1. 부모 클래스(COMBDataset) 초기화 (데이터 로딩 준비 완료)
#         super().__init__(config, filepath, **kwargs)

#         # [핵심] 첫 번째 파일의 info를 읽어서 '이름 -> 인덱스' 변환
#         if self.drop_channels:
#             try:
#                 # 첫 번째 파일 경로를 이용해 _info.pkl 찾기
#                 sample_file = self.file_paths[0]
#                 info_path = sample_file.replace('.npy', '_info.pkl')
                
#                 with open(info_path, 'rb') as f:
#                     info = pickle.load(f)
#                     all_ch_names = info['ch_names'] if isinstance(info, dict) else info.ch_names
                
#                 # 대소문자 무시하고 매칭하기 위해 upper() 변환 비교 추천
#                 drop_channels_upper = [n.upper() for n in self.drop_channels]
#                 all_ch_names_upper = [n.upper() for n in all_ch_names]

#                 # 인덱스 찾기
#                 self.drop_indices = [
#                     i for i, name in enumerate(all_ch_names_upper) 
#                     if name in drop_channels_upper
#                 ]
                
#                 print(f"[*] Channel Drop Init: {self.drop_channels} -> Indices {self.drop_indices}")

#                 total_ch = len(all_ch_names)
#                 self.keep_indices = [i for i in range(total_ch) if i not in self.drop_indices]
#                 # 모델 입력 차원 수정
#                 self.model_input_ch = len(self.keep_indices)

#             except Exception as e:
#                 print(f"[Warning] Failed to load info for channel drop: {e}")
#                 print("Channel drop will be skipped.")
#                 self.drop_indices = []

#     def __getitem__(self, index):
#         # 1. 부모 클래스의 __getitem__ 호출 -> 이미 전처리/정규화/텐서변환 된 데이터를 받음
#         # 반환값: X(Tensor), Y(Tensor), stat, mask
#         # X shape: (N_trials, N_channels, N_times)
#         X, Y, stat, mask = super().__getitem__(index)

#         # 2. Channel Drop 로직 적용
#         if self.drop_channels:
#             X = X[:, self.keep_indices, :]
                
#             # 주의: stat이나 mask도 채널 차원이 있다면 같이 줄여줘야 함
#             # stat shape: (N_trials, N_channels, 2) -> 이것도 줄여야 함
#             if stat.shape[1] > len(self.keep_indices):
#                 stat = stat[:, self.keep_indices, :]
#                 mask = mask[:, self.keep_indices, :]

#         return X, Y, stat, mask
    
#     def _get_sample_info(self):
#         total_time, ch = super()._get_sample_info()
#         return total_time, ch-len(self.drop_channels)

#region [!] MODEL DESIGN
#SECTION - MODEL DESIGN
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)    # Score every patches

    def forward(self, x, mask=None):
        # x:[Batch, Patch_Num, embed_dim*embed_num]

        # 1. Score
        scores = self.attention(x).squeeze(-1) #[Match, Patch_Num]

        # 2. masking(option: ignore padding part)
        if mask is not None:
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask==0, min_value)



        # 3. calculate the weight (Softmax)
        weights = F.softmax(scores, dim=-1) #[Batch, Patch_Num] -> sum is 1

        # 4. weighted sum 
        # combine patches multiflied weight
        # bmm(batch matrix multiplication)
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1) #[Batch, 512]

        return x, weights 



class LitEEGPTCausal_LoRA(pl.LightningModule):       # !) Transformer(encoder -> decoder) structure -> classifier(encoder -> Linear(Flatten))

    def __init__(self, config,
                 target_class_idx =None,        # if None, multiple classification, Or if number, class number or others
                 fixed_train_patch_idx = None,  # if None, None/random, if number, specific patch idx
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()  
        self.save_hyperparameters()
        self.evaluator = Evaluator(config)
        self.visualizer = Visualizer(config)
        self.metrics_list = config["metrics"]

        #--------EEG data part----------------
        self.input_ch = config["input_ch"]
        self.input_len = config["input_len"]
        self.sampling_rate = config["sampling_rate"]
        self.total_sec = 1
        self.target_class_idx=target_class_idx

        # # --------masking part----------------
        # self.target_class_idx = target_class_idx
        self.fixed_train_patch_idx = fixed_train_patch_idx

        self.time_bin = config["time_bin"]   


        #---------classification part---------
        if self.target_class_idx is None:
            self.num_class = config["n_classes"]
            print(f"[*] Mode: Multi-class Classification(5 classes with null)")  
        else:
            self.num_class = 1
            print(f"[*] Mode: Binary Classification (class {target_class_idx} vs Others)")


        # ----------training part-------------
        self.embed_dim = 512
        self.embed_num = 4
        self.encoder_out_dim = self.embed_dim * self.embed_num

        # init model
        target_encoder = EEGTransformer(
            img_size=[self.input_ch, self.input_len],          # ? 256hz * 30 sec = num of datapoint
            patch_size=32*2,                        
            patch_stride = 8,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        use_channels_names =[k.upper() for v,k in enumerate( [          # 57-4 = 53ch #drop_channels = ['TP9', 'TP10', 'FT9', 'FT10']
                                    'Fpz',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
            'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                        'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                            'O1', 'O2', 'Oz'])]
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        self.valid_auroc = BinaryAUROC()
        self.valid_f1 = BinaryF1Score()

        # -- load checkpoint
        if load_path and os.path.exists(load_path):
            pretrain_ckpt = torch.load(load_path)
            
            target_encoder_stat = {}
            for k,v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]]=v
            
                    
            self.target_encoder.load_state_dict(target_encoder_stat)
        else:
            print("[*] Skipped loading checkpoint(Init blank model)")

        self.chan_conv       = Conv1dWithConstraint(self.input_ch, self.input_ch, 1, max_norm=1)
        
        # -------------LoRA---------------------------
        # CHANGE target_modules

        ##FIXME - R, target_modules
        # [Case A] Cross Subject
        r_custom = 8   # normal
        lora_custom = 32
        target_modules_custom = ["qkv", "fc1", "fc2"] # "proj":attn makes better acc

        # [Case B] Within Subject
        # r_custom = 4     # if data is small
        # lora_custom = 64
        # target_modules_custom = ["q_proj", "v_proj"] -> must modify
        
        peft_config = LoraConfig(
            inference_mode = False,
            r = r_custom,                             # Rank: for Cross-Subject Generalization
            lora_alpha = lora_custom,
            lora_dropout =0.1,
            target_modules = target_modules_custom,
            bias="none"
        )

        # Encoder -> LoRA
        # Weights of self.target_encoder are Freezed automatically
        self.target_encoder = get_peft_model(self.target_encoder, peft_config)

        print(">>> LoRa Applied to Encoder! Trainable Params:")
        self.target_encoder.print_trainable_parameters()

        # 1) weighted pooling
        self.pooler = AttentionPooling(embed_dim=self.encoder_out_dim)

        # !) MLP Head instead of Decoder
        self.head = torch.nn.Sequential(
                                        torch.nn.BatchNorm1d(self.encoder_out_dim),
                                        torch.nn.Linear(self.encoder_out_dim,512),
                                        torch.nn.GELU(),
                                        torch.nn.Dropout(0.5),
                                        torch.nn.Linear(512,128),
                                        torch.nn.GELU(),
                                        torch.nn.Dropout(0.5),                                        
                                        # LinearWithConstraint(128, self.num_class, max_norm = 0.5)
                                        torch.nn.Linear(128, self.num_class)
                                        )

        # !) debug for training 
        # !!!!) BatchNorm is key point because data is so small, 
        # self.head = nn.Linear(self.encoder_out_dim,self.num_class)

        # self.head = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(self.encoder_out_dim),
        #     torch.nn.Linear(self.encoder_out_dim, self.num_class)
        # )
        if self.target_class_idx is not None:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0)) #pos_weight = unbalanced weight
            # BCE contains Sigmoid(this is useful for binary)
            # unvalance problem <- pos weight (1 vs 5)


        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True

        # !) Debug with Freezing encoder parameter
        # we will freeze until train few epochs
        # for param in self.target_encoder.parameters():
        #     param.requires_grad=False

        #Head and Pooling keep train
        

        
    def forward(self, x, mask = None):
        # 1. Data processing
        # B, C, T = x.shape   # C=53, T=256
        x = self.chan_conv(x)

        # 2. pass through Encoder
        # ?) shape of z = [Batch_size, Patch_num, 512] 
        # = # z shape: [Batch, 97, 512] (N=97는 stride=2 때문에 생긴 결과)
        # self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x), mask_x=None)
        # print(f"z mean: {z.mean().item()}, z std: {z.std().item()}") # should be not 0


        ''' 3-1. CLS token Cross Attention
        # copy CLS token as like batch size
        cls_token = self.cls_token.expand(B, -1, -1)    # This is query

        h, _ = self.cross_attention(query=cls_token, key=z, value=z)
        h = h.squeeze(1)
        '''
    
        # if len(z.shape) == 4:
        #     z=z.flatten(2)

        # --------------------------------
        #          Masking Part
        # --------------------------------
        B, N, C, D = z.shape #[8, 21, 4, 512]
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.float()
            patch_mask = F.adaptive_max_pool1d(mask, output_size = N)
            patch_mask, _ = patch_mask.max(dim=1)
            patch_mask = patch_mask
        else:
            patch_mask = None   

        # 3. Flatten
        # h = z.flatten(2)
        # h = z.mean(dim=1)
        z_flattened = z.flatten(2)  #[B, N, C*D]
        h, attn_weight = self.pooler(z_flattened,mask=patch_mask)
        # print(f"h mean:{h.mean().item()}")  # chekc if it is NaN or 0

        # 4. classification(MLP method)
        h = self.head(h)

        # x is raw data for logging, h is prediction
        return x, h

    def training_step(self, batch):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, _, mask= batch
        
        
        if self.target_class_idx is not None:   #(One-ve-Rest)
            label=(y==self.target_class_idx).float().unsqueeze(1)   #[Batch, 1]
        else:
            label = y.long()

        
        x, logit = self.forward(x,mask=mask)
        loss = self.loss_fn(logit, label)
        
        if self.target_class_idx is  None:
            accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()

            # Logging to TensorBoard by default
            self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
            self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
            self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
            self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
            self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)

        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        # 1. Sanity Check
        # if self.is_sanity:
        #     self.is_sanity = False
        #     self.running_scores["valid"] = []  # empty  
        #     return super().on_validation_epoch_end()
        
        # 2. Collect Data(List -> Tensor)
        # self.running_scores["valid"]-> (label, logit)
        if not self.running_scores["valid"]: return #If there's not data
        
        label_list = [l.to(self.device) for l, s in self.running_scores["valid"]]
        score_list = [s.to(self.device) for l, s in self.running_scores["valid"]]

        labels = torch.cat(label_list, dim=0) 
        logits = torch.cat(score_list, dim=0)

        # 2-1. Distributed Gathering
        labels = self.all_gather(labels) 
        logits = self.all_gather(logits)
        
        # gather ���� ������ [GPU��, Batch, ...] �� ���������� ������ ��
        # if labels.ndim > 1: labels = labels.flatten(0, 1)
        # if logits.ndim > 1: logits = logits.flatten(0, 1)

        labels = labels.cpu().numpy() # Metric ������ Numpy
        logits = logits.float()
        # ---------------------------------------------------------
        # 3. Mode Selection & Prediction (Binary vs Multi-class)
        # ---------------------------------------------------------+

        best_threshold = 0.5 #default

        # [Class A] One-vs-Rest(Binary)
        if self.target_class_idx is not None:
            # 3-1. Logit -> Probability Transform
            #Binary logit -> Sigmoid > threshold -> 0 or 1
            probs = torch.sigmoid(logits).cpu()
            labels = labels.astype(int)
            preds = (probs >= best_threshold).long().numpy()  # default threshold=0.5
            probs = probs.numpy()
        # [Case B] Multi-class
        else:
            # 3-1. Prediction(Argmax)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            print(f"DEBUG: {labels.shape}, {preds.shape}, {probs.shape}")
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        print(f"DEBUG: {labels.shape}, {preds.shape}, {probs.shape}")
        metric_results = self.evaluator.compute_metrics(
            preds=preds, 
            labels=labels, 
            probs=probs, 
            params=self.metrics_list 
        )
        for metric_name, value in metric_results.items():
            self.log(f"valid_{metric_name}", value, on_epoch=True, prog_bar=True)

        # 4. Confusion Matrix
        should_log_image = (self.current_epoch + 1) % 10 == 0
        if should_log_image and isinstance(self.logger, WandbLogger) and self.global_rank == 0:
            #(N,) vs (N,) For Safty
            # if preds.ndim >1: preds = preds.squeeze()
            # if labels.ndim >1: labels = labels.squeeze()
            fig = self.visualizer.plot_cm(config["save_dir"],  labels, preds, 
                                    config["patch_idx"], config["patch_idx"], 
                                    metric_results['acc'], epoch=self.current_epoch+1)
            self.logger.experiment.log({
                        "valid/confusion_matrix_img": wandb.Image(fig),
                        "global_step": self.global_step
                    })
        self.running_scores["valid"] = []
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # print(f"DEBUG: {self.target_class_idx}")
        x, y, _, mask = batch
        x, logit = self.forward(x, mask=mask)

        if self.target_class_idx is not None:   # Binary (One-vs-Rest)
            label = (y == self.target_class_idx).float().unsqueeze(1)  # [Batch, 1]
            loss = self.loss_fn(logit, label) 

            self.log('valid_auroc', self.valid_auroc(torch.sigmoid(logit.view(-1)), label.view(-1).long()), 
                     on_epoch=True, prog_bar=True)
            self.log('valid_f1', self.valid_f1(torch.sigmoid(logit.view(-1)), label.view(-1).long()), 
                     on_epoch=True, prog_bar=True)
        else:
            label = y.long()

            loss = self.loss_fn(logit, label)

            accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
            # Logging to TensorBoard by default
            # self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
            
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        
        return loss
    
    def configure_optimizers(self):
        
        ''' decoder version
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())
            +list(self.linear_probe1.parameters())
            +list(self.linear_probe2.parameters())
            +[self.cls_token]
            +list(self.decoder.parameters())
            ,weight_decay=0.01)


        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        '''
        # steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        # steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        encoder_lora_params = filter(lambda p: p.requires_grad, self.target_encoder.parameters())
        
        optimizer = torch.optim.AdamW([
            {   # <group1>: Encoder, it's already smart, so train slowly
                'params':encoder_lora_params,
                'lr':3e-4
            },
            {   # <group2>: others(Head, Pooler, Conv) -> Faster
                'params': list(self.head.parameters()) +
                          list(self.pooler.parameters())+
                          list(self.chan_conv.parameters()),
                'lr':3e-3
            }
        ], weight_decay= 0.05)
        
        total_steps = self.trainer.estimated_stepping_batches

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = [3e-4, 3e-3],
            total_steps = total_steps,
            pct_start = 0.2
        )
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': 'learning_rate', # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
#!SECTION
#endregion [!] MODEL DESIGN

#region [!] MAIN
##SECTION - MAIN
# %%MAIN
if __name__ == "__main__":

    import sys

    # [1] 디버거가 붙어있는지 확인하는 함수
    def is_debugging():
        gettrace = getattr(sys, 'gettrace', None)
        # VS Code debugpy나 일반 디버거가 활성화 상태면 True 반환
        if gettrace is None:
            return False
        return gettrace() is not None

    # [2] 상황에 따른 전략 설정
    if is_debugging():
        print("🚀 [Debug Mode Detected] Switching to Single GPU (strategy='auto')")
        # 디버깅 중일 땐: DDP 끄고, GPU 1개만 사용 (그래야 안 멈춤)
        strategy = 'auto' 
        devices = [0]     
        num_workers = 1   # 디버깅 땐 0이 안전함
    else:
        print("⚡ [Training Mode] Using DDP Strategy")
        # 평소 실행(python ~)일 땐: 원래대로 DDP 사용
        strategy = 'ddp'
        devices = 'auto' # 또는 [0, 1]
        num_workers = 16  # 원래 설정
    IS_DEBUG = False if not is_debugging() else True
    config = {
                "data_dir": "./EEG(256Hz)_COMB/",
                "batch_size": 64,
                "accumulate_grad_batches" : 1,
                "num_workers": num_workers,
                "shuffle": True,
                "sampling_rate": 256,
                "start_time_ms" : -200,
                "data_ext": "npy",
                "window_type": "fixed",  # "fixed" or "random"
                "time_bin": 16,
                "file_chunk_type": "subject", # "subject" or "run"
                "normalize_method": "zscore", # "zscore" or "minmax"
                "patch_idx": None,
                "stride": None,
                "num_epochs": 100 if not IS_DEBUG else 2,
                "patience": 10,
                "n_classes": 6,
                "is_label_null": True,
                "skip_pred" : False,
                "metrics":["acc", "bal_acc"],
                "save_dir": None,
                "ckpt_dir": None,
                "csv_input_dir":None,
                "test_files":None,
        }



    # The order of EEG Data from Combine Lab




    #FIXME - DEBUG
    # IS_DEBUG = True
    target_cls = 2

    k_folds = 2
    # k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # ------------------config---------------------------

    train_dir = os.path.join(config["data_dir"], "processed_train/npy")
    test_dir = os.path.join(config["data_dir"], "processed_test/npy")

    train_files = [
            os.path.join(train_dir, f) for f in os.listdir(train_dir) 
            if "label" not in f and "stats" not in f and "info" not in f
        ]
    
    test_files = [
            os.path.join(test_dir, f) for f in os.listdir(test_dir) 
            if "label" not in f and "stats" not in f and "info" not in f
        ]
    
    config["test_files"] = test_files
    input_len, input_ch = COMBDataset(config=config, filepath = train_files)._get_sample_info()
    n_patches = input_len//config["time_bin"]

    config["n_patches"] = n_patches
    config["input_len"] = input_len
    config["input_ch"] = input_ch

    train_date = datetime.now().strftime('%Y%m%d_%H%M')

    # 폴드 결과 저장
    fold_results = []

    # -------------------------------------------------------
    # Loop 1: K fold
    # -------------------------------------------------------
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_files)):
    # for fold in range(1):
        
        current_fold = fold + 1
        print(f"\n" + "="*40)
        print(f"[:] Starting Fold {current_fold}/{k_folds}")
        print("="*40)


        # -------------------------------------------------------
        # Loop 2: Patches (For Each Time Bin)
        # -------------------------------------------------------
        #FIXME - DEBUG(patch)
        for patch_idx in range(1, n_patches):
                
            train_files_fold = [train_files[i] for i in train_ids]
            val_files_fold = [train_files[i] for i in val_ids]
            
            target_cls = 'All'
            ##FIXME - CONFIG(project name)
            # ------------------config---------------------------
            experiment_name = f"LORA_F{current_fold}_P{patch_idx}_C{target_cls}"
            exp_id = f"{train_date}_{experiment_name}"
            print(f"[*] Experiment ID: {exp_id}")
            print(f"[*] Mode: {'DEBUG (Overfit 1 batch)' if IS_DEBUG else 'TRAINING'}")

            base_dir = f"./EEGPT/logs/{exp_id}/"
            ckpt_dir = os.path.join(base_dir,"checkpoints")
            analysis_dir = os.path.join(base_dir,"analysis")

            config["save_dir"] = analysis_dir
            config["ckpt_dir"] = ckpt_dir

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(analysis_dir, exist_ok=True)

            print(f"  Training Class:{target_cls}")
            print(f"  Patch Index: {patch_idx}")

            config["patch_idx"] = patch_idx

            train_dataset = COMBDataset(config=config, filepath = train_files_fold)
            train_loader = torch.utils.data.DataLoader(
                                                train_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=config["shuffle"], 
                                                num_workers=config["num_workers"],
                                                collate_fn = torch_collate_fn,)

            val_dataset = COMBDataset(config=config, filepath = val_files_fold)
            val_loader = torch.utils.data.DataLoader(
                                                val_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                num_workers=config["num_workers"],
                                                collate_fn = torch_collate_fn,
                                                persistent_workers=True)

            #ANCHOR - Training
            # -- begin Training ------------------------------

            torch.set_float32_matmul_precision('medium' )

            # 4-3. init model(MultiClass)
            model = LitEEGPTCausal_LoRA(
                config = config,
                fixed_train_patch_idx=patch_idx, 
            )
            # print(model)
            wandb_logger = WandbLogger(
                                project="eegpt_combine3_LoRa_util", 
                                name=experiment_name,
                                group= experiment_name, 
                                id = exp_id,
                                tags=[f"Fold{experiment_name}",f"Patch{patch_idx}",f"Class{target_cls}","Masking", exp_id, f"Timebin{config['time_bin']}"],
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

            early_stop_callback = EarlyStopping(
                monitor = 'valid_loss',
                min_delta = 0.00,
                patience = 5,
                verbose=True,
                mode='min'
            )
                        
            callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

        
            ''' 
            !) pl.Trainer is for LightningModule 
            
            1. manage hardware => accelerator
            2. optimization => Mixed Precision
            3. Manage Train loop => max_epoch
            4. Logging => logger
            5. checkpoint, learning Rate monitor -> callback
            '''
            #FIXME - Trainer
            trainer = pl.Trainer(accelerator='cuda',
                                devices=devices,
                                strategy=strategy,
                                sync_batchnorm=True,
                                precision=16,
                                max_epochs=config["num_epochs"], 
                                accumulate_grad_batches = config["accumulate_grad_batches"],
                                callbacks=callbacks,
                                enable_progress_bar=True,
                                num_sanity_val_steps=0,
                                check_val_every_n_epoch=1 if IS_DEBUG else 10,
                                limit_val_batches=0.25,
                                logger=[wandb_logger, 
                                        pl_loggers.CSVLogger(base_dir, name="EEGPT_COMBINE_csv")])



            trainer.fit(model, train_loader, val_loader)

            wandb.finish()

            wandb_logger.experiment.finish()
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
                
        # -------------------------------------------------------
        # END Loop 2: Patches (For Each Time Bin)
        # -------------------------------------------------------
    if not IS_DEBUG and fold_results:
        print(f"\n[!] {k_folds}-Fold CV Finished.")
        print(f"Average Val Loss: {np.mean(fold_results):.4f}")

    # # 4. Best Model Inference

    # if not IS_DEBUG: # INFERENCE
    # best_ckpt_path = trainer.checkpoint_callback.best_model_path
    # print(f"Best Model saved at: {best_ckpt_path}")

    # # 3. OverLoad Best Model Checkpoint
    # best_checkpoint = torch.load(best_ckpt_path)
    # model.load_state_dict(best_checkpoint['state_dict']) 

    # model.eval()
    # model.cuda()
    # manager = InferenceManager(config, model = LitEEGPTCausal_LoRA)
    # manager.run()
    

#!SECTION
#endregion [!] MAIN