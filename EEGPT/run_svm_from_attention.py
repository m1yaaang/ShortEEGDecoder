

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
import shutil

from sklearn.decomposition import PCA
from joblib import Parallel, delayed


from finetune_EEGPT_combine import (
    LitEEGPTCausal, 
    COMBLoader, 
    prepare_COMB_dataset
)

def plot_tgm_matrix(tgm_matrix, sampling_rate, min_len, save_path):
    plt.figure(figsize=(8, 7))
    
    # Generate time axis label
    ticks = np.arange(0, tgm_matrix.shape[0], 2) # interval = 2
    tick_labels = [f"{(t * min_len)/sampling_rate:.2f}s" for t in ticks]
    
    # Heatmap

    img = plt.imshow(tgm_matrix, interpolation='nearest', origin='lower', 
                     cmap='viridis', vmin=0.15, vmax=0.3)
    # origin='lower'-> (0,0) locates lower left
    
    plt.xticks(ticks, tick_labels, rotation=45)
    plt.yticks(ticks, tick_labels)
    
    plt.xlabel("Test Time (s)")
    plt.ylabel("Train Time (s)")
    plt.title("Temporal Generalization Matrix")
    plt.colorbar(label="Accuracy")
    
    # Diagonal) - matched time
    plt.plot([0, tgm_matrix.shape[1]-1], [0, tgm_matrix.shape[0]-1], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] TGM Plot saved to {save_path}")

    # [Key] Upload image to WandB
    if wandb.run is not None:
        wandb.log({"TGM/Matrix_Heatmap": wandb.Image(save_path, caption="Temporal Generalization Matrix")})
        print("[*] Plot uploaded to WandB!")

def save_tgm_text_results(tgm_matrix, sampling_rate, min_len, save_dir):
    # 1. Save CSV (Full Matrix)
    csv_path = os.path.join(save_dir, "tgm_matrix_raw.csv")
    df = pd.DataFrame(tgm_matrix)
    
    # Add time information to columns/index
    time_labels = [f"{(t * min_len)/sampling_rate:.3f}s" for t in range(tgm_matrix.shape[0])]
    df.columns = time_labels
    df.index = time_labels
    df.to_csv(csv_path)
    
    # 2. Extract Diagonal Performance (Test on same time segment)
    diag_acc = np.diag(tgm_matrix)

    time_steps = np.arange(len(diag_acc)) * min_len / sampling_rate
    
    # 3. Generate Summary Text
    summary_text = "\n[TGM Summary Report]\n"
    summary_text += f"Avg Matrix Accuracy: {np.mean(tgm_matrix):.4f}\n"
    summary_text += f"Avg Diagonal Accuracy: {np.mean(diag_acc):.4f}\n"
    summary_text += f"Max Diagonal Accuracy: {np.max(diag_acc):.4f} at {time_labels[np.argmax(diag_acc)]}\n"
    summary_text += "-"*30 + "\n"
    summary_text += "Time (s) | Diagonal Acc\n"
    
    for t, acc in zip(time_labels, diag_acc):
        summary_text += f"{t:8} | {acc:.4f}\n"
        
    print(summary_text) # Print to console
    
    # Save text file
    txt_path = os.path.join(save_dir, "tgm_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary_text)
        
    print(f"[*] Text results saved: {csv_path}, {txt_path}")

    # Upload Raw Data and Summary to WandB
    if wandb.run is not None:

        data = [[t, acc] for (t, acc) in zip(time_steps, diag_acc)]
        table = wandb.Table(data=data, columns=["Time_sec", "Accuracy"])
        
        # (2) Line Plot 
        # X: Time_sec, Y: Accuracy
        wandb.log({
            "TGM/Diagonal_Line_Chart": wandb.plot.line(
                table, 
                "Time_sec", 
                "Accuracy", 
                title="Diagonal Accuracy over Time"
            )
        })

        max_idx = np.argmax(diag_acc)
        wandb.log({
            "Summary/Max_Accuracy": np.max(diag_acc),
            "Summary/Time_at_Max_Acc": time_steps[max_idx],
            "Summary/Mean_Accuracy": np.mean(diag_acc)
        })


def log_diagonal_plot_with_wandb(tgm_matrix, sampling_rate, min_len, save_dir):
    """
    Extracts diagonal accuracy, plots the graph, highlights the maximum point, and uploads it to WandB.
    Usage: log_diagonal_plot_with_wandb(tgm_mat, 256, m_len, save_root)
    """
    # 1. Extract data
    diag_acc = np.diag(tgm_matrix)
    times = np.arange(len(diag_acc)) * min_len / sampling_rate
    
    # 2. Calculate maximum point
    max_idx = np.argmax(diag_acc)
    max_time = times[max_idx]
    max_acc = diag_acc[max_idx]

    # 3. Plot graph
    plt.figure(figsize=(10, 5))
    plt.plot(times, diag_acc, label='Accuracy', color='#1f77b4')
    
    # Mark the maximum point (Red dot & Text)
    plt.scatter(max_time, max_acc, color='red', s=50, zorder=5)
    plt.annotate(f"Max: {max_acc:.1%} @ {max_time:.2f}s", 
                 xy=(max_time, max_acc), xytext=(0, 10), textcoords='offset points',
                 ha='center', color='red', fontweight='bold')

    # Set style
    plt.title("Diagonal Accuracy (Max Point Highlighted)")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 4. Save and Upload to WandB
    save_path = os.path.join(save_dir, "diagonal_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    if wandb.run is not None:
        # [Key] Upload image in a single line
        wandb.log({"TGM/Diagonal_Analysis": wandb.Image(save_path)})
        
        # (Optional) Log max value separately to display at the top of the dashboard
        wandb.log({"Max_Diagonal_Acc": max_acc, "Max_Time_Sec": max_time})


def get_time_config(model):
    sampling_rate = model.sampling_rate
    min_len = int(sampling_rate * 0.05)
    model_input_len = 256
    total_blocks = model_input_len // min_len
    
    return min_len, total_blocks


def load_model_from_checkpoint(ckpt_path, device='cuda'):
    print(f"[*] Loading model from {ckpt_path}...")
    
    # 1. init model(same options as finetuning)
    # load_path: pre-train weights, finetuning ckpt: load from checkpoint
    model = LitEEGPTCausal(load_path="") 
    
    # 2. load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # 3. overload weights
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model


def extract_features(model, loader, device, total_blocks, min_len):
    """
    Save every feature vector(h) and label(y) from all data
    output shape: [total_samples, total_blocks, Feature_dim]
    """

    model.eval()
    model.to(device)

    all_features = [] #[batch, time, dim]
    all_labels = []

    # Freeze Encoder
    for param in model.parameters():
        param.requires_grad = False

    stride = model.target_encoder.patch_embed.patch_stride or model.target_encoder.patch_embed.patch_size
    patch_size = model.target_encoder.patch_embed.patch_size

    # Cache pre-calculated mask indices (Speed optimization)
    # Calculate once instead of every batch
    mask_indices = []
    for t in range(total_blocks):
        target_start = t * min_len
        target_end = (t + 1) * min_len
        mask_indices.append({'start': target_start, 'end': target_end})

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extraction"):
            x, y, _ = batch
            x = x.to(device)
            
            # Encoder
            x_conv = model.chan_conv(x)
            z_full = model.target_encoder(x_conv, model.chans_id.to(x), mask_x=None)
            
            B, N, E, D = z_full.shape
            z_full = z_full.reshape(B, N, -1) # (Batch, N, Dim)
            
            batch_time_features = []

            # Block Loop (Minimize mask generation inside loop for optimization)
            for t in range(total_blocks):
                t_info = mask_indices[t]
                
                # Generate Mask (Batch operation)
                # Simplified logic to reduce overhead of creating torch.zeros every time
                # Only find overlapping patch indices and pool corresponding z vectors
                
                temp_mask = torch.zeros((B, N), device=device)
                active = False
                
                for p_idx in range(N):
                    p_start = p_idx * stride
                    p_end = p_start + patch_size
                    if (p_end > t_info['start']) and (p_start < t_info['end']):
                        temp_mask[:, p_idx] = 1.0
                        active = True
                
                if active:
                    h, _ = model.pooler(z_full, mask=temp_mask)
                else:
                    h = torch.zeros((B, model.encoder_out_dim), device=device)
                
                batch_time_features.append(h.cpu().numpy())
            
            # Stack: (Batch, Time, Dim)
            batch_time_features = np.stack(batch_time_features, axis=1)
            all_features.append(batch_time_features)
            all_labels.append(y.cpu().numpy())

    X_all = np.concatenate(all_features, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    return X_all, y_all

def run_svm_from_attention(model, train_loader, test_loader, device="cuda", save_dir = './', mode="train", infer_feature="./"):
    min_len, total_blocks = get_time_config(model)
    
    expected_dim = model.encoder_out_dim

    #---- feature vector cashing -----
    # 1. Feature Extraction(TrainSet/TestSet split)
    feature_save_path = os.path.join(save_dir, "extracted_features.npz")
    if mode != "train":
        print(f"[*] Found saved features at {infer_feature}. Loading...")
        data = np.load(infer_feature)
        X_train_seq = data['X_train']
        y_train = data['y_train']
        X_test_seq = data['X_test']
        y_test = data['y_test']
        print("[*] Features loaded successfully!")
    else:
        print("[*] No saved features found. Starting Extraction (This takes time)...")
        print("--- Extracting Train Set ---")
        X_train_seq, y_train = extract_features(model, train_loader, device, total_blocks, min_len)
        
        print("--- Extracting Test Set ---")
        X_test_seq, y_test = extract_features(model, test_loader, device, total_blocks, min_len)
        
        # extracted Feature 
        print(f"[*] Saving extracted features to {feature_save_path}...")
        np.savez_compressed(
            feature_save_path, 
            X_train=X_train_seq, 
            y_train=y_train, 
            X_test=X_test_seq, 
            y_test=y_test
        )

    svm_model_dir = os.path.join(save_dir, "svm_models")
    os.makedirs(svm_model_dir, exist_ok=True)

    # ----------------------PCA----------------------
    print("[*] Applying PCA to reduce dimensions...")

    # data -> (samples*Times, Dim)
    n_train_samples ,n_times, n_features = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(-1, n_features)

    # keep dimension with 99% of variance  
    pca = PCA(n_components=0.99, random_state=42)
    X_train_flat = pca.fit_transform(X_train_flat)

    # Reshape
    new_dim = X_train_flat.shape[1]
    X_train_seq = X_train_flat.reshape(n_train_samples, n_times, new_dim)

    n_test_samples = X_test_seq.shape[0]
    X_test_flat = X_test_seq.reshape(-1, n_features)
    
    X_test_flat = pca.transform(X_test_flat)
    X_test_seq = X_test_flat.reshape(n_test_samples, n_times, new_dim)
    
    print(f"[*] Dimension Reduced: {n_features} -> {new_dim} (Keep 99% Variance)")
    # ----------------------PCA----------------------


    # 2. TGM Loop
    # Matrix init (Train_Time * Test_Time)
    tgm_matrix = np.zeros((total_blocks, total_blocks))

    print(f"[*] Starting TGM Loop({total_blocks}*{total_blocks}...)")

    '''
    # Train Time Loop(t)
    for t_train in tqdm.tqdm(range(total_blocks),desc="TGM Rows(Train Time)"):
        
        # get data of time step t
        X_tr_t = X_train_seq[:, t_train, :] #[Samples, Dim]

        # classifier train (SVM)
        # better converge with pipeline standardScaler
        clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))
        clf.fit(X_tr_t, y_train)

        joblib.dump(clf, os.path.join(svm_model_dir, f"svm_t{t_train}.pkl"))

        #Test Time Loop(t')
        for t_test in range(total_blocks):
            # t' data
            X_te_t = X_test_seq[:, t_test, :]

            #score
            acc = clf.score(X_te_t, y_test)
            tgm_matrix[t_train, t_test] = acc
    '''

    
    def process_one_timepoint(t_train):
        
        X_tr_t = X_train_seq[:, t_train, :]
        
        # SVM train
        # max_iter limit (1000 -> 2000 safety)
        clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42, max_iter=2000))
        clf.fit(X_tr_t, y_train)
        
        row_scores = []
        for t_test in range(total_blocks):
            X_te_t = X_test_seq[:, t_test, :]
            score = clf.score(X_te_t, y_test)
            row_scores.append(score)
            
        return row_scores

    # joblib:parallel (n_jobs=-1: Use every CPU cores)
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_one_timepoint)(t) for t in range(total_blocks)
    )

    for t_train, row_scores in enumerate(results):
        tgm_matrix[t_train, :] = row_scores    

    return tgm_matrix, total_blocks, min_len






if __name__ == "__main__":


    start_time = datetime.now().strftime('%Y%m%d_%H%M')
    save_root = f"./svm_from_attention/{start_time}_TGM_Analysis"
    os.makedirs(save_root, exist_ok=True)

    # CKPT_PATH = "./eegpt_combine/20251211_1531_patchsize=12_1/checkpoints/best-epoch=04-valid_loss=1.6087.ckpt" 
    CKPT_PATH = "logs/final_model_epoch100.ckpt"
    DATA_ROOT = "./EEG/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

    run = wandb.init(
        project="eegpt_svm_combine", 
        name = "svm_pca_from_attention",
        job_type="test",
        tags=["test", "heatmap", "tgm", "svm"]
    )

    # 1. prepare test data (Use only Test Set)
    # prepare_COMB_dataset or mask only TestLoader
    print("[*] Preparing Test Data...")
    train_dataset, test_dataset, valid_dataset = prepare_COMB_dataset(DATA_ROOT)
    
    # Batch Size is influenced by the memory size of the GPU(can be bigger if needed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

    # 2. load Model
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")
        
    model = load_model_from_checkpoint(CKPT_PATH, device=DEVICE)
    #model = LitEEGPTCausal.load_from_checkpoint(CKPT_PATH) # it could be easier but not recommended

    tgm_mat, blocks, m_len = run_svm_from_attention(
        model, 
        train_loader, 
        test_loader, 
        device=DEVICE,
        save_dir=save_root,
        mode="infer",
        infer_feature="./svm_from_attention/20251212_1734_TGM_Analysis/extracted_features.npz"
    )

    plot_tgm_matrix(tgm_mat, 256, m_len, "tgm_svm_from_attention.png")

    save_tgm_text_results(tgm_mat, 256, m_len, save_root)

    log_diagonal_plot_with_wandb(tgm_mat, 256, m_len, save_root)

    print(f"\n[*] Analysis Finished! All results saved in: {save_root}")
    wandb.finish()