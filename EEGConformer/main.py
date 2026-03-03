import torch
import torch.nn as nn
import numpy as np

from braindecode import EEGConformer


class EEGConformer_Trainer:
    def __init__(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

class EEGConformer_Inferencer:
    def __init__(self):
        pass

    def infer(self):
        pass

class EEGConformer_Visualizer:
    def __init__(self):
        pass

    def plot(self):
        pass

model = EEGConformer(
    n_outputs=6,
    n_chans=53,
    filter_time_length=16,
    pool_time_stride=4,
    sfreq=500,
)

if __name__ == "__main__":
    from plot_tgm_per_epoch import main as plot_tgm_main
    from plot_best_epoch_diagonal import main as plot_diag_main

    print("[*] Starting TGM per epoch plotting...")
    plot_tgm_main()

    print("\n[*] Starting best-epoch diagonal plotting...")
    plot_diag_main()

    print("\n[!] All done.")

