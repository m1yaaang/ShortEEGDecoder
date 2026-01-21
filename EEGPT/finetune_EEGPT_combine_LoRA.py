'torchrun --nproc_per_node=2 finetune_EEGPT_combine_LoRA.py'

import os

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
matplotlib.use('Agg')

from utils import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from peft import LoraConfig, get_peft_model
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
'''
# Common constants
#FIXME - CONFIG
MODEL_INPUT_LEN = 256

#region [!] LOAD DATA
#SECTION - LOAD DATA
class COMBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=256, is_train=True, is_augment = False, patch_idx = None, time_bin= None):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate
        self.is_train = is_train
        self.is_augment = is_augment
        self.patch_idx = patch_idx

        self.model_input_len = MODEL_INPUT_LEN
        if time_bin is None:
            self.time_bin = int(self.sampling_rate*0.05)
        else:
            self.time_bin = time_bin


        self.total_blocks = self.model_input_len//self.time_bin

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"] - 1)        #[2, 3, 4, 5, 6] -> [0 1 2 3 4]
        
        model_input_len = self.model_input_len
        data_len = X.shape[-1]

        # z-score noramlization
        mean = np.mean(X, axis=-1,keepdims=True)
        std=np.std(X, axis=-1, keepdims=True)+1e-6
        X = (X-mean)/std

        time_bin = self.time_bin
        total_blocks = self.total_blocks

        # total_patch_N = model_input_len //time_bin
        max_valid_patch = data_len // time_bin



        if self.is_augment and data_len > time_bin:
            if random.random() < 0.5:
                #cumulative Window Augmentation
                end_patch_idx = random.randint(1,max_valid_patch)
                start_patch_idx = 0
            else:
                # random window augmentation
                end_patch_idx = random.randint(1,max_valid_patch)
                start_patch_idx = random.randint(0,end_patch_idx)
        else:   
            #[Test]
            start_patch_idx = 0
            end_patch_idx = max_valid_patch


        input_tensor = torch.zeros((X.shape[0], model_input_len))
        mask = torch.zeros(model_input_len)


        input_tensor[:,start_patch_idx:min(time_bin*end_patch_idx, data_len)] = torch.from_numpy(X[:,start_patch_idx:(time_bin*end_patch_idx)])

        mask[start_patch_idx:(time_bin*end_patch_idx)] = 1.0
        return input_tensor, Y, mask
    

def prepare_COMB_dataset(root, time_bin):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = COMBLoader(
        os.path.join(
            root, "processed_train"), train_files, is_train=True, shuffle=True, time_bin=time_bin
    )
    test_dataset = COMBLoader(
        os.path.join(
            root, "processed_test"), test_files, is_train=False, shuffle=True, time_bin=time_bin
    )
    # val_dataset = COMBLoader(
    #     os.path.join(
    #         root, "processed_eval"), val_files, is_train=False, time_bin=time_bin
    # )
    val_dataset = None
    print(f"COMB) train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")



    x,y,mask = train_dataset[0]
    print(f"dataset size: {x.size()}")
    return train_dataset, test_dataset, val_dataset

def get_dataset(args):
    # Data is prcessed by make_COMB.py
    if args.dataset == 'COMB':
        train_dataset, test_dataset, val_dataset = prepare_COMB_dataset("./EEG/")
        ch_names = [k.upper() for v,k in enumerate( [          # 57-4 = 53ch #drop_channels = ['TP9', 'TP10', 'FT9', 'FT10']
                                'Fpz',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                        'O1', 'O2', 'Oz'])]
        args.nb_classes = 5
        metrics = ["accuracy"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics

#!SECTION
#endregion [!] LOAD DATA

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


# The order of EEG Data from Combine Lab
use_channels_names =[k.upper() for v,k in enumerate( [          # 57-4 = 53ch #drop_channels = ['TP9', 'TP10', 'FT9', 'FT10']
                                'Fpz',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                        'O1', 'O2', 'Oz'])]




class LitEEGPTCausal_LoRA(pl.LightningModule):       # !) Transformer(encoder -> decoder) structure -> classifier(encoder -> Linear(Flatten))

    def __init__(self, 
                 target_class_idx =None,        # if None, multiple classification, Or if number, class number or others
                 fixed_train_patch_idx = None,  # if None, None/random, if number, specific patch idx
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()  
        self.save_hyperparameters()
        
        #--------EEG data part----------------
        self.chans_num = len(use_channels_names)
        self.sampling_rate = 256
        self.total_sec = 1
        self.model_input_len = MODEL_INPUT_LEN

        # --------masking part----------------
        self.target_class_idx = target_class_idx
        self.fixed_train_patch_idx = fixed_train_patch_idx

        self.time_bin = int(self.sampling_rate*0.05)     
        '''
        patch size
        sampling rate *0.1 = 25 = 100ms
        sampling rate *0.05 = 12 = 48ms

        '''

        #---------classification part---------
        if self.target_class_idx is None:
            self.num_class = 6
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
            img_size=[self.chans_num, self.sampling_rate * self.total_sec],          # ? 256hz * 30 sec = num of datapoint
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

        self.chan_conv       = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        ''' Decoder part 
        self.linear_probe1   = LinearWithConstraint(2048, 64, max_norm=1)
        self.drop            = torch.nn.Dropout(p=0.50)        
        self.decoder         = torch.nn.TransformerDecoder(
                                    decoder_layer=torch.nn.TransformerDecoderLayer(64, 4, 64*4, activation=torch.nn.functional.gelu, batch_first=False),
                                    num_layers=4
                                )
        self.cls_token =        torch.nn.Parameter(torch.rand(1,1,64)*0.001, requires_grad=True)
        self.linear_probe2   =   LinearWithConstraint(64, self.num_class, max_norm=0.25)
        '''

        '''
        # 1-1) CLS token
        # standard initiation method from BERT, ViT
        self.cls_token = torch.nn.Parameter(torch.zeros(1,1,512))
        nn.init.normal_(self.cls_token, std=0.02)

        # Q:CLS token, K/V: ourput of encoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=512,num_head=8, batch_first=True)
        '''

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
        B, C, T = x.shape   # C=53, T=256
        x = self.chan_conv(x)

        # 2. pass through Encoder
        # ?) shape of z = [Batch_size, Patch_num, 512] 
        # = # z shape: [Batch, 97, 512] (N=97는 stride=2 때문에 생긴 결과)
        # self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x), mask_x=None)
        # print(f"z mean: {z.mean().item()}, z std: {z.std().item()}") # should be not 0

        ''' Decoder part
        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)
        
        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h], dim=1)
        h = h.transpose(0,1)
        h = self.decoder(h, h)[0,:,:]
        
        h = self.linear_probe2(h)
        '''

        ''' 3-1. CLS token Cross Attention
        # copy CLS token as like batch size
        cls_token = self.cls_token.expand(B, -1, -1)    # This is query

        h, _ = self.cross_attention(query=cls_token, key=z, value=z)
        h = h.squeeze(1)
        '''
    
        if len(z.shape) == 4:
            z=z.flatten(2)

        # --------------------------------
        #          Masking Part
        # --------------------------------


        self.total_blocks = self.model_input_len//self.time_bin

        if self.training and (self.fixed_train_patch_idx is not None):
            B, N, _ = z.shape
            time_mask = torch.zeros((B, self.model_input_len)).to(z.device)
            t_start = (self.fixed_train_patch_idx)*self.time_bin
            t_end = (self.fixed_train_patch_idx+1)*self.time_bin
            # patch_mask = torch.zeros((B,N)).to(z.device)
            
            if t_start < self.model_input_len: 
                time_mask[:, t_start:min(t_end, self.model_input_len)] = 1.0
            patch_mask = F.adaptive_max_pool1d(time_mask.unsqueeze(1),output_size=z.shape[1]).squeeze(1)
            if patch_mask.sum() == 0:
                print("!!!! WARNING: ALL MASKS ARE ZERO !!!!")
        else:
            if mask is not None:
                patch_mask = F.adaptive_max_pool1d(mask.unsqueeze(1),output_size=z.shape[1]).squeeze(1)
                # save this mask 
                # [1, 1, 1, 0, 0] -> 1    
                if patch_mask.sum() == 0:
                    print("!!!! WARNING: ALL MASKS ARE ZERO !!!!")
            else:
                patch_mask = None

        

        # 3. Flatten
        # h = z.flatten(2)
        # h = z.mean(dim=1)
        h, attn_weight = self.pooler(z,mask=patch_mask)
        # print(f"h mean:{h.mean().item()}")  # chekc if it is NaN or 0

        # 4. classification(MLP method)
        h = self.head(h)

        # x is raw data for logging, h is prediction
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, mask= batch
        
        
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
        if not self.running_scores["valid"]:
            return #If there's not data
        
        label_list, score_list = [],[]
        for l, s in self.running_scores["valid"]:
            label_list.append(l)
            score_list.append(s)

        label = torch.cat(label_list, dim=0).cpu()      #lable (0, 1, 2,..)
        y_score = torch.cat(score_list, dim=0).cpu()    #Output(logit)

        # ---------------------------------------------------------
        # 3. Mode Selection & Prediction (Binary vs Multi-class)
        # ---------------------------------------------------------+

        best_threshold = 0.5 #default

        # [Class A] One-vs-Rest(Binary)
        if self.target_class_idx is not None:
            # 3-1. Logit -> Probability Transform
            #Binary logit -> Sigmoid > threshold -> 0 or 1
            probs = torch.sigmoid(y_score.float())
            y_true = label.float()

            #(N,1) -> (N,)
            if preds.dim() >1: preds = preds.squeeze()
            if y_true.dim() >1: y_true = y_true.squeeze()

            #3-2. Find Best Threshold(Precision-Recall Curve)
            precision, recall, thresholds = precision_recall_curve(y_true.numpy(), probs.numpy())

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            print(f"[Epoch {self.current_epoch}] Best Threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")

            # 3-3. Prediction(Best Threshold)
            preds = (probs > best_threshold).long()

            class_names = ["Rest", f"Target({self.target_class_idx})"]
            model_output_for_metric = probs.numpy()
            is_binary_mode = True

        # [Case B] Multi-class
        else:
            # 3-1. Prediction(Argmax)
            preds = torch.argmax(y_score, dim=-1)
            class_names = [str(i) for i in range(len(torch.unique(label)))]

            model_output_for_metric = torch.softmax(y_score.float(), dim=-1).numpy()
            is_binary_mode = False

        # 4. Confusion Matrix
        should_log_image = (self.current_epoch + 1) % 10 == 0 or (self.current_epoch == self.trainer.max_epochs - 1)
        if should_log_image:
            #(N,) vs (N,) For Safty
            if preds.dim() >1: preds = preds.squeeze()
            if label.dim() >1: label = label.squeeze()

            if len(label) == len(preds):
                cm = confusion_matrix(label.long().numpy(), preds.numpy())

                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix(Epoch{self.current_epoch})')

                # 5. Image log on WandB
                if isinstance(self.logger, WandbLogger):
                    self.logger.experiment.log({
                        "valid/confusion_matrix": wandb.Image(fig),
                        "global_step":self.global_step
                    })

                plt.close(fig)

        # 6. Metric
       
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(
            output=model_output_for_metric, 
            target=label.numpy(), 
            metrics=metrics, 
            is_binary=is_binary_mode,  # True/False 
            threshold=best_threshold   
        )
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)

        self.running_scores["valid"] = []
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # print(f"DEBUG: {self.target_class_idx}")
        x, y, _ = batch
        x, logit = self.forward(x)

        if self.target_class_idx is not None:   # Binary (One-vs-Rest)
            label = (y == self.target_class_idx).float().unsqueeze(1)  # [Batch, 1]

            label_for_metric = label.squeeze() 
            logit_for_metric = logit.squeeze()
            if self.target_class_idx is not None:
                loss = self.loss_fn(logit, label) 

            self.log('valid_auroc', self.valid_auroc(torch.sigmoid(logit_for_metric), label_for_metric), 
                     on_epoch=True, prog_bar=True)
            self.log('valid_f1', self.valid_f1(torch.sigmoid(logit_for_metric), label_for_metric), 
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
                'lr':1e-4
            },
            {   # <group2>: others(Head, Pooler, Conv) -> Faster
                'params': list(self.head.parameters()) +
                          list(self.pooler.parameters())+
                          list(self.chan_conv.parameters()),
                'lr':1e-3
            }
        ], weight_decay= 0.05)
        
        total_steps = self.trainer.estimated_stepping_batches

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = [1e-4, 1e-3],
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
    target_cls = 2
    train_patch_idx = [9, 10, 11, 12, 13, 14, 15]
    # train_patch_idx = [7]

    ##FIXME - CONFIG(experiment setting)
    # ------------------config---------------------------
    time_bin = 16   # 16 timepoints = 62.5ms/ 64 timepoints = 250ms
    # batch_size = 8*4
    batch_size = 32  # 메모리 부족 문제로 줄임 (기존 32)
    accumulate_grad_batches = 4  # batch_size 줄인만큼 늘림 (기존 4)
    max_epochs = 100 if not IS_DEBUG else 2 # 디버그 시 에포크 단축

    # Classes: Original labels are 1-6, after (label-1) in dataset: 0-5                                                   
    # Class 0 (original 1) = Null -> always "Rest"           
    # Classes 1-5 (original 2-6) = Targets for OVR           
    TARGET_CLASSES = [1, 2, 3, 4, 5]  # After label-1 transformation   
    # TOTAL_PATCHES=[256//time_bin]

    k_folds = 2
    # k_folds = 5
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
        # for patch_idx in range(TOTAL_PATCHES):
        for patch_idx in train_patch_idx:
                if patch_idx ==5 or patch_idx ==6: continue
            # -------------------------------------------------------
            # Loop 3: Target Class (OVR Binary Classifiers)
            # -------------------------------------------------------    
            #FIXME - DEBUG(Target Class)        
            # for target_cls in TARGET_CLASSES:
            # for target_cls in [target_cls]:
                target_cls = 'All'
                ##FIXME - CONFIG(project name)
                # ------------------config---------------------------
                experiment_name = f"LORA_F{current_fold}_P{patch_idx}_C{target_cls}"
                exp_id = f"{train_date}_{experiment_name}"
                print(f"[*] Experiment ID: {exp_id}")
                print(f"[*] Mode: {'DEBUG (Overfit 1 batch)' if IS_DEBUG else 'TRAINING'}")

                base_dir = f"./logs/{exp_id}/"
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
                model = LitEEGPTCausal_LoRA(
                    fixed_train_patch_idx=patch_idx
                )
                # print(model)
                wandb_logger = WandbLogger(
                                    project="eegpt_combine3_LoRa", 
                                    name=experiment_name,
                                    group= experiment_name, 
                                    id = exp_id,
                                    tags=[f"Fold{experiment_name}",f"Patch{patch_idx}",f"Class{target_cls}","Masking", exp_id, f"Timebin{time_bin}"],
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
                debug = None
                trainer = pl.Trainer(accelerator='cuda',
                                    devices=[0,1],
                                    precision=16,
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
    if not IS_DEBUG and fold_results:
        print(f"\n[!] {k_folds}-Fold CV Finished.")
        print(f"Average Val Loss: {np.mean(fold_results):.4f}")

#!SECTION
#endregion [!] MAIN