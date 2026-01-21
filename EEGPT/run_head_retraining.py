import copy
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import torch
from tqdm import tqdm

def get_time_config(model):
    sampling_rate = model.sampling_rate
    min_len = int(sampling_rate * 0.05)
    model_input_len = 256
    total_blocks = model_input_len // min_len
    
    return min_len, total_blocks

def reset_head_weights(layer):
    """init Head Layer weight of model"""
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

def train_head_for_tgm(model, train_loader, mask_generator_func, t_idx, device, epochs=3):
    """
    quickly train Head for specific time(t_idx)
    """
    # Encoder Freeze, Head : train
    model.target_encoder.eval()
    model.pooler.eval()
    model.head.train()
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for batch in train_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            # Encoder (No Grad)
            with torch.no_grad():
                x_conv = model.chan_conv(x)
                z_full = model.target_encoder(x_conv, model.chans_id.to(x), mask_x=None)
                
                # make mask for t_idx, Pooling
                temp_mask = mask_generator_func(x, z_full, t_idx, device)
                h, _ = model.pooler(z_full, mask=temp_mask)
            
            # Head (Train)
            optimizer.zero_grad()
            logits = model.head(h)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

def evaluate_head_for_tgm(model, test_loader, mask_generator_func, t_idx, device):
    """Eval head for specific time(t_idx)"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            x_conv = model.chan_conv(x)
            z_full = model.target_encoder(x_conv, model.chans_id.to(x), mask_x=None)
            
            temp_mask = mask_generator_func(x, z_full, t_idx, device)
            h, _ = model.pooler(z_full, mask=temp_mask)
            
            logits = model.head(h)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
    return correct / total

def run_head_retraining(model, train_loader, test_loader, device="cuda"):
    min_len, total_blocks = get_time_config(model)
    tgm_matrix = np.zeros((total_blocks, total_blocks))
    
    # generate mask helper
    def generate_mask(x, z_full, t_idx, dev):
        B, N, _, _ = z_full.shape # if z_full is 4D
        # B, N, D = z_full.shape # if z_full is 3D
        if len(z_full.shape) == 3: B, N, _ = z_full.shape

        target_start = t_idx * min_len
        target_end = (t_idx + 1) * min_len
        
        stride = model.target_encoder.patch_embed.patch_stride or model.target_encoder.patch_embed.patch_size
        patch_size = model.target_encoder.patch_embed.patch_size
        
        mask = torch.zeros((B, N)).to(dev)
        for p in range(N):
            p_s = p * stride
            p_e = p_s + patch_size
            if (p_e > target_start) and (p_s < target_end):
                mask[:, p] = 1.0
        return mask

    print(f"[*] Starting Rapid Retraining TGM ({total_blocks}x{total_blocks})...")
    
    # Train Loop (Rows)
    for t_train in tqdm(range(total_blocks), desc="Training Heads"):
        
        # 1. Head init (reset weight)
        model.head.apply(reset_head_weights)
        
        # 2.  Head train with at t_train data(3 Epoch)
        train_head_for_tgm(model, train_loader, generate_mask, t_train, device, epochs=3)
        
        # Test Loop (Cols)
        for t_test in range(total_blocks):
            # 3. Eval t_test data 
            acc = evaluate_head_for_tgm(model, test_loader, generate_mask, t_test, device)
            tgm_matrix[t_train, t_test] = acc
            
    return tgm_matrix, total_blocks, min_len

if __name__ == "__main__":
    tgm_mat, blocks, m_len = run_head_retraining(model, train_loader, test_loader)
    plot_tgm_matrix(tgm_mat, 256, m_len, "tgm_strategy2.png")
