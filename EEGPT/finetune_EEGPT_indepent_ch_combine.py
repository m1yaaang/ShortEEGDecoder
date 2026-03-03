import random 
import os
from types import NoneType
import torch
from torch import nn
import pytorch_lightning as pl
import pickle

import torchvision
from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F


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

from utils_my import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Modules.models.EEGPT_mcae import EEGTransformer
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

# original code
#use_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ' ]
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
class COMBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=256,is_train=True):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate
        self.is_train = is_train

        self.model_input_len = 256
        self.min_len = int(self.sampling_rate*0.05)

        self.total_blocks = self.model_input_len//self.min_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"] - 2)        #[2, 3, 4, 5, 6] -> [0 1 2 3 4]
        
        model_input_len = 256
        data_len = X.shape[-1]

        # z-score noramlization
        mean = np.mean(X, axis=-1,keepdims=True)
        std=np.std(X, axis=-1, keepdims=True)+1e-6
        X = (X-mean)/std

        min_len = self.min_len
        total_blocks = self.total_blocks

        # total_patch_N = model_input_len //min_len
        max_valid_patch = data_len // min_len



        if self.is_train and data_len > min_len:
            if random.random() < 0.5:
                #cumulative Window Augmentation
                start_patch_idx = 0
                end_patch_idx = random.randint(1,max_valid_patch)
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


        input_tensor[:,start_patch_idx:min(min_len*end_patch_idx, data_len)] = torch.from_numpy(X[:,start_patch_idx:(min_len*end_patch_idx)])

        mask[start_patch_idx:(min_len*end_patch_idx)] = 1.0
        return input_tensor, Y, mask
    

def prepare_COMB_dataset(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = COMBLoader(
        os.path.join(
            root, "processed_train"), train_files, is_train=True
    )
    test_dataset = COMBLoader(
        os.path.join(
            root, "processed_test"), test_files, is_train=False
    )
    val_dataset = COMBLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_train=False
    )
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




class LitEEGPTCausal(pl.LightningModule):       # !) Transformer(encoder -> decoder) structure -> classifier(encoder -> Linear(Flatten))

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        self.num_class = 5
        self.sampling_rate = 256
        self.total_sec = 1

        self.embed_dim = 512
        self.embed_num = 4
        self.encoder_out_dim = self.embed_dim * self.embed_num

        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, self.sampling_rate * self.total_sec],          # ? 256hz * 30 sec = num of datapoint
            patch_size=32*2,                        
            patch_stride = 16,
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

        self.chan_conv       = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1, groups = self.chans_num)
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

        self.loss_fn        = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True

        # !) Debug with Freezing encoder parameter
        # we will freeze until train few epochs
        # for param in self.target_encoder.parameters():
        #     param.requires_grad=False

        #Head and Pooling keep train
        

        ## Encoder -> 1ch
        #
        print(f"[*] modifyying Encoder Config for Channel Independence:{self.target_encoder.num_patches} -> (1,13)")

        # 1. num_patch modify (53, 13) -> (1, 13)
        original_time_patches = self.target_encoder.num_patches[1]

        new_patches_tuple = (1, original_time_patches)
        self.target_encoder.num_patches = (1, original_time_patches)
        if hasattr(self.target_encoder, 'patch_embed'):
            print(f"[*] Modifying patch_embed config to: {new_patches_tuple}")
            self.target_encoder.patch_embed.num_patches = new_patches_tuple
            
            # Remove img_size from patch_embed if it exists to prevent potential conflicts
            if hasattr(self.target_encoder.patch_embed, 'img_size'):
                 del self.target_encoder.patch_embed.img_size
        
        # Remove img_size from target_encoder if it exists
        if hasattr(self.target_encoder, 'img_size'):
             del self.target_encoder.img_size
        print(f"[*] Modification Complete. Encoder num_patches: {self.target_encoder.num_patches}")


        
    def forward(self, x, mask = None):
        # 1. Data processing
        B, C, T = x.shape
        x = temporal_interpolation(x, self.sampling_rate * self.total_sec)   #256hz * 30sec

        # conv1d: channel mixing, do depthwise or pass identity
        x = self.chan_conv(x)

        # 2. pass through Encoder
        # ?) shape of z = [Batch_size, Patch_num, 512]
        # self.target_encoder.eval()
        # z = self.target_encoder(x, self.chans_id.to(x), mask_x=None)
        # print(f"z mean: {z.mean().item()}, z std: {z.std().item()}") # should be not 0

        # ================= [Modification 3] Batch Folding Logic =================
        
        # A. Flatten Channels into Batch dimension
        # [B, C, T] -> [B*C, 1, T]
        x_flat = x.reshape(B * C, 1, -1)

        # B. Prepare Spatial Embeddings (Channel IDs)
        # We repeat the channel IDs for each batch sample.
        # self.chans_id: [C] -> view(-1) -> [C]
        # unsqueeze(0) -> [1, C]
        # expand(B, -1) -> [B, C] (Efficient copying)
        # reshape(-1) -> [B*C]
        ids_flat = self.chans_id.view(-1).unsqueeze(0).expand(B, -1).reshape(-1)
        ids_flat = ids_flat.to(x.device)

        # C. Encoder Pass (Channel Independent)
        # The encoder sees (B*C) samples, each with 1 channel.
        # However, 'ids_flat' provides the spatial identity for each sample.
        # z_flat output: [B*C, Patch_Num, Dim]
        z_flat = self.target_encoder(x_flat, ids_flat, mask_x=None)

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
    
        # 3. Mask Handling (Expand mask for B*C)
        if mask is not None:
            # mask: [B, T] -> Need to match patch dimension first
            # We assume the mask is for the 'Time' dimension.
            # Using adaptive_max_pool to downsample mask to Patch_Num (N)
            _, N, D = z_flat.shape
            
            # [B, T] -> [B, 1, N]
            patch_mask = F.adaptive_max_pool1d(mask.unsqueeze(1), output_size=N)
            
            # Expand to matches [B, C, N] then flatten to [B*C, N]
            # [B, 1, N] -> [B, C, N] -> [B*C, N]
            patch_mask_flat = patch_mask.expand(B, C, N).reshape(B * C, N)
            
            # Safety check
            if patch_mask_flat.sum() == 0:
                print("!!!! WARNING: ALL MASKS ARE ZERO !!!!")
        else:
            patch_mask_flat = None

        # 4. Attention Pooling (Independent per channel)
        # z_flat is already [B*C, N, D], so we can pass it directly to pooler
        # h_flat: [B*C, D]
        h_flat, attn_weight = self.pooler(z_flat, mask=patch_mask_flat)
        
        # 5. Restore Dimensions & Flatten
        # [B*C, D] -> [B, C, D]
        h = h_flat.reshape(B, C, -1)
        
        # Flatten for MLP Head: [B, C*D]
        h = h.flatten(1)

        # 6. Classification Head
        logit = self.head(h)

        return x, logit
        

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, mask= batch
        label = y.long()
        
        x, logit = self.forward(x,mask=mask)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        

        if batch_idx ==0:
            loss.backward(retain_graph=True)

            # print("\n"+"="*30)

            head_grad = self.head[1].weight.grad if isinstance(self.head, nn.Sequential) else self.head.weight.grad
            # print(f"1. Head Gradient:{head_grad.abs().mean().item() if head_grad is not None else 'None(broken!)'}")

            self.log('Head Gradient',head_grad.abs().mean().item(), on_epoch=True, on_step=False, sync_dist=True)
            enc_grad = list(self.target_encoder.parameters())[-1].grad

            # print(f"2. Encoder Gradient:{enc_grad.abs().mean().item() if enc_grad is not None else 'None(Frozen/broken!)'}")
            self.log('Head Gradient',enc_grad.abs().mean().item(), on_epoch=True, on_step=False, sync_dist=True)
            # print("\n"+"="*30)


        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        
        return super().on_validation_epoch_end()
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, _ = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
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

        optimizer = torch.optim.AdamW([
            {   # <group1>: Encoder, it's already smart, so train slowly
                'params':self.target_encoder.parameters(),
                'lr':1e-4
            },
            {   # <group2>: others(Head, Pooler, Conv) -> Faster
                'params': list(self.head.parameters()) +
                          list(self.pooler.parameters())+
                          list(self.chan_conv.parameters()),
                'lr':1e-3
            }
        ], weight_decay= 0.05)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = [1e-4, 1e-3],
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
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



def analyze_temporal_importance(model, test_loader, mode="Test", device="cuda",save_dir="./logs/temporal"):
    model.eval()
    model.to(device)

    if model.target_encoder.patch_embed.patch_stride is not None:
        stride = model.target_encoder.patch_embed.patch_stride
    else:
        stride = model.target_encoder.patch_embed.patch_size
        
    patch_size = model.target_encoder.patch_embed.patch_size    
    sampling_rate = model.sampling_rate # 256Hz 

    dataset = test_loader.dataset

    min_len = dataset.min_len       # 12=0.04sec
    total_blocks = dataset.total_blocks # 21

    print(f"[*] Analysis Config: Stride={stride}, Hz={sampling_rate}, patch_length = {min_len}, total_blocks = {total_blocks}")

    # load patch info
    block_accuracies = {}
    global_accuracies = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Analyzing Patches"):
            x,y,_= batch # ignore original mask
            x= x.to(device)
            y= y.to(device)

            #1. Figure out patch num through encoder
            x_conv = model.chan_conv(x)
            z_full = model.target_encoder(x_conv, model.chans_id.to(x), mask_x=None)


            # if len(z_full.shape) == 4:
            #     z_full=z_full.flatten(2)

            B, N, E, D = z_full.shape
            z_full = z_full.reshape(B, N, -1)
            num_patches = z_full.shape[1]

            # 1. Global Performance Check(without masking)
            h_global, _ = model.pooler(z_full,mask=None)
            logits_global = model.head(h_global)
            pred_global = torch.argmax(logits_global, dim=-1)
            acc_global = (pred_global==y).float().mean().item()
            global_accuracies.append(acc_global)

            '''
            # 2. eval every patches (slicing Window analysis)
            for i in range(num_patches):
                if i not in patch_accuracies: patch_accuracies[i] = []

                # -- Fake mask set
                #" i'th patch = 1, the others = 0"
                temp_mask = torch.zeros((x.shape[0],num_patches)).to(device)
                temp_mask[:, i] = 1.0

                # -- put this mask into attention pooling
                # model try to classify with this ith patche
                h, _ = model.pooler(z_full, mask=temp_mask)
                logits = model.head(h)

                #save result
                pred = torch.argmax(logits, dim=-1)
                acc = (pred==y).float().mean().item()
                patch_accuracies[i].append(acc)
            results = []

            print("\n[Temporal Analysis Result]")
            for i, accs in block_accuracies.items():
                avg_acc= np.mean(accs)

                time_sec =(i*stride)/float(sampling_rate)
                print(f"Time{time_sec:.3f}s (Patch{i}):Acc ={avg_acc*100:.2f}%")

                results.append({"patch_idx":i,
                                "time_sec":time_sec,
                                "accuuracy":avg_acc})
            '''
            
            # 2. eval every time block(0.04sec)
            for block_idx in range(total_blocks):
                if block_idx not in block_accuracies:
                    block_accuracies[block_idx] = []
                
        
                #(1) view point index
                target_start = block_idx * min_len
                target_end = (block_idx+1)*min_len

                #(2) overlap patches
                temp_mask = torch.zeros((B,N)).to(device)

                #(3) patch size(64) is bigger than block(12), can be overlapped multiple patches
                active_count = 0
                for p_idx in range(N):
                    p_start = p_idx*stride
                    p_end = p_start + patch_size

                    # overlap check

                    if (p_end>target_start) and (p_start < target_end):
                        temp_mask[:, p_idx] = 1.0
                        active_count += 1

                # Eval, if there is any patches overlapped
                if active_count > 0:
                    h,_ = model.pooler(z_full, mask=temp_mask)
                    logits = model.head(h)
                    pred = torch.argmax(logits, dim=-1)
                    acc = (pred ==y).float().mean().item()
                else:   # impossible this 
                    acc = -1

                block_accuracies[block_idx].append(acc)

        total_global_acc = np.mean(global_accuracies)
        print(f"Total Global Accuracy: {total_global_acc*100:.2f}%")

        if total_global_acc < 0.25:
            print("!!!! WARNING: Global Accuracy is too low !!!!")
        
        results = []
        print("\n[Segment-wise Temporal Analysis Result]")

        for block_idx in sorted(block_accuracies.keys()):
            avg_acc = np.mean(block_accuracies[block_idx])

            time_start_sec = (block_idx * min_len) / float(sampling_rate)
            time_end_sec = (( block_idx + 1)*min_len) / float(sampling_rate)

            print(f"Time {time_start_sec:.3f}~{time_end_sec:.3f}s: Acc={avg_acc*100:.2f}%")

            results.append({
                "block_idx":block_idx,
                "time_start": time_start_sec,
                "time_end": time_end_sec,
                "accuracy": avg_acc,
                "mode":mode
            })


        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(f"{save_dir}/temporal_result_{mode}.csv", index=False)

        accuracies = df['accuracy'].values.reshape(1,-1)
        times= df['time_start'].values

        plt.figure(figsize=(12,2))

        img = plt.imshow(accuracies, cmap='BrBG', aspect='auto', vmin=0, vmax=0.3)

        # plt.yticks([])
        tick_interval = max(1, len(times)//10)
        plt.xticks(
            ticks=np.arange(len(times))[::tick_interval],
            labels=[f"{t:.2f}s" for t in times[::tick_interval]],
            fontsize = 9
            )
        plt.xlabel("Time (s)")
        plt.title(f"{mode} Temporal Importance Heatmap(Stride = {stride})")
        
        cbar = plt.colorbar(img, fraction=0.046, pad = 0.04)
        cbar.set_label("Accuracy")

        plt. tight_layout()

        img_path = os.path.join(save_dir,f"heatmap_{mode}.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        


        if wandb.run is not None:
            
            table = wandb.Table(dataframe=df)
        
            
            wandb.log({
                f"{mode}/Temporal Analysis":wandb.plot.line(table,"time_sec","accuracy"),
                f"{mode}/Temporal Analysis(Heatmap)":wandb.Image(img_path, caption="Time-resolved Accuracy"),
                f"{mode}/Accuracy Line": wandb.plot.line(
                    wandb.Table(dataframe=df), "time_start", "accuracy",title=f"{mode} Accuracy"
                )
                })
    return df



#
if __name__ == "__main__":
    # load configs
    from datetime import datetime
    # -- LOSO 
    # Train Data Num : 5Class: 2 3 4 5 6
    subjects = [22]
    N = len(subjects)//10
    set_all = set(subjects)

    for fold in range(10):
    # for fold in range(1):


        set_valid = set(subjects[fold*N:(fold+1)*N])
        set_train = set_all - set_valid


        print(f"Fold {fold+1} Train: {set_train}, Valid: {set_valid}")

        train_dataset, test_dataset ,valid_dataset = prepare_COMB_dataset("./EEG/")

    # ------------------config---------------------------

        experiment_name = "chans_indepent_patchsize=12"
        exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{experiment_name}"

        base_dir = f"./logs/{exp_id}_Fold{fold+1}"
        ckpt_dir = os.path.join(base_dir,"checkpoints")
        analysis_dir = os.path.join(base_dir,"analysis")

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)

        wandb_logger = WandbLogger(
                            project="eegpt_combine", 
                            name=f"Fold_{fold+1}_{experiment_name}",
                            group=exp_id,
                            id=f"{exp_id}_{fold+1}",
                            tags=[experiment_name,],
                            save_dir=base_dir
                            )

        # wandb_logger.experiment.config.update({"max_epochs": max_epochs})
    # -------------------config--------------------------


        # -- begin Training ------------------------------

        import math
        torch.set_float32_matmul_precision('medium' )


        batch_size=8*4

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
        test_lodaer = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)


        max_epochs = 150
        # steps_per_epoch = math.ceil(len(train_loader))
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4

        # init model
        model = LitEEGPTCausal()

        wandb_logger.watch(model, log="all", log_graph=True, log_freq=100)
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

  
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_loss',
            mode='min',
            save_top_k=5,
            save_last=True,
            filename='best-{epoch:02d}-{valid_loss:.4f}',
            verbose=True,
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
        # trainer = pl.Trainer(accelerator='cuda',
        #                     precision=16,
        #                     max_epochs=max_epochs, 
        #                     callbacks=callbacks,
        #                     logger=[wandb_logger, 
        #                             pl_loggers.CSVLogger(base_dir, name="EEGPT_COMBINE_csv")])

        ##For Debug Dataset and network
        # trainer=pl.Trainer(accelerator='gpu',
        #                     devices=[0],
        #                     precision=16,
        #                     max_epochs=1,
        #                     limit_train_batches=1,
        #                     limit_val_batches=0,
        #                     num_sanity_val_steps=0,
        #                     logger=False,
        #                     enable_checkpointing=False)

        ## For Debug Overfit one Batch
        trainer = pl.Trainer(
            accelerator='gpu', devices=[0],
            max_epochs=40,
            precision=16,
            num_sanity_val_steps=0,
            overfit_batches=1,
            logger=[wandb_logger, pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_COMBINE_tb", version=f"fold{fold+1}"), 
                    pl_loggers.CSVLogger('./logs/', name="EEGPT_COMBINE_csv")]
        )

        trainer.fit(model, train_loader, valid_loader)

        trainer.save_checkpoint(f"{ckpt_dir}/final_model_epoch100.ckpt")

        analyze_temporal_importance(model, valid_loader, save_dir=analysis_dir, mode="valid")
        analyze_temporal_importance(model, test_lodaer, save_dir=analysis_dir)