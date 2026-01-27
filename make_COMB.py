import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

from tqdm import tqdm

'''
/home/winter/eegdecoder/minyung/EEGPT/downstream_combine/run_class_finetuning_EEGPT_change_comb.py
    split and edit in code 217
'''

def get_file_paths(subject_id, run_n, data_dir):
    stim_file_path = os.path.join(
                data_dir,
                f"{subject_id}",
                "TRI",
                "epochs_data",
                f"{subject_id}_run{run_n}_stim_icaRej-epo.fif"
            )

    iti_file_path = os.path.join(
                data_dir,
                f"{subject_id}",
                "TRI",
                "epochs_data",
                f"{subject_id}_run{run_n}_iti_icaRej-epo.fif"
        )
    path = {
        "stim": stim_file_path,
        "iti": iti_file_path
    }
    return path


def read_sample_info(subject_ids, data_dir):
    """Reads information from a sample .fif file."""
    sample_file = os.path.join(
        data_dir,
        f"{subject_ids[0]}",
        "TRI",
        "epochs_data",
        f"{subject_ids[0]}_run01_stim_icaRej-epo.fif"
    )
    epochs = mne.read_epochs(sample_file, preload=True, verbose=False)
    
    # ch_names = epochs.info['ch_names']
    # sampling_rate = epochs.info['sfreq']

    # result = {
    #     "ch_names": ch_names,
    #     "sampling_rate": sampling_rate
    # }

    return epochs.info

def save_metadata(info, subject_id, data_dir):      # mne info object
    save_path = os.path.join(data_dir, f"{subject_id}_info.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(info, f)
        
    # print(f"Full MNE Info object saved to {save_path}")

def read_mne_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    epochs = mne.read_epochs(file_path, preload=True, verbose=False)

    return epochs

def preprocess_epochs(epochs, target_sr = None, drop_chs = None, ch_order = None):
    if target_sr != epochs.info['sfreq'] and target_sr is not None:
        epochs.resample(sfreq=target_sr)

    if drop_chs is not None:
        useless_chs = []
        for ch in drop_chs:
            if ch in epochs.info['ch_names']:
                useless_chs.append(ch)
        epochs.drop_channels(useless_chs)
    if ch_order is not None and len(ch_order) == len(epochs.info['ch_names']):
        epochs.reorder_channels(ch_order)
    # Iterate through each trial within the Epochs object.

    return epochs

def extract_stim(epochs):

    # Extract data and event labels into NumPy arrays.
    X = epochs.get_data()      # Shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]    # Shape: (n_trials,)
    
    X_list = []
    y_list = []
    # print(X.shape)
    for i in range(len(X)):
        single_trial_data = X[i]
        single_trial_label = y[i]-1

        X_list.append(single_trial_data)
        y_list.append(single_trial_label)

    return np.array(X_list), np.array(y_list), epochs.info['ch_names']


def extract_iti(epochs, null_label=0):

    # Extract data and event labels into NumPy arrays.
    X = epochs.get_data()      # Shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]    # Shape: (n_trials,)
    
    X_list = []
    y_list = []

    # print(X.shape)
    n_trials = X.shape[0]
    n_keep = int(n_trials * 0.2)  # 1/5
    selected_idx = np.random.choice(n_trials, size=n_keep, replace=False)
    for i in selected_idx:
        single_trial_data = X[i]
        # null data should be 1
        # interval time(1s) between objects 
        single_trial_label = null_label
        
        X_list.append(single_trial_data)
        y_list.append(single_trial_label)

    return np.array(X_list), np.array(y_list), epochs.info['ch_names']


def export_np(subject_id, run_n, X_all, y_all, ch_names, save_dir, file_ext="npy"):
    """
    X_subj: (sum_trials, n_channels, n_times)
    y_subj: (sum_trials,)
    
    if npz, data_structure: [subject] = [[x_subj,y_subj], [x_subj,y_subj], [x_subj,y_subj], ...]
    """

    mean = np.mean(X_all, axis=-1, keepdims=True)
    std = np.std(X_all, axis=-1, keepdims=True)

    # 통계량을 하나의 배열로 결합 (Mean, Std 포함) -> (N_trials, n_channels, [mean, std])
    stats = np.concatenate([mean, std], axis=-1)

    save_dir = os.path.join(save_dir, file_ext)
    os.makedirs(save_dir, exist_ok=True)

    if run_n is not None:
        file_name = f"{subject_id}_run{run_n}"
    else:
        file_name = f"{subject_id}"


    if file_ext == "npy":
        np.save(os.path.join(save_dir, f"{file_name}.npy"), X_all)
        np.save(os.path.join(save_dir, f"{file_name}_label.npy"), y_all)
        np.save(os.path.join(save_dir, f"{file_name}_stats.npy"), stats)

    elif file_ext == "npz":
        np.savez_compressed(os.path.join(save_dir, f"{file_name}.npz"), X=X_all, y=y_all, ch_names=ch_names, stats=stats)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    # print(f"{subject_id}: X={X_all.shape}, y={y_all.shape}, y unique={np.unique(y_all)}")


def export_pkl(subject_id, run_n, X_all, y_all, ch_names, save_dir):
    """
    X_subj: (sum_trials, n_channels, n_times)
    y_subj: (sum_trials,)
    stats: (sum_trials, n_channels, 2)  # Mean and Std
    """

    mean = np.mean(X_all, axis=-1, keepdims=True)
    std = np.std(X_all, axis=-1, keepdims=True)

    # 통계량을 하나의 배열로 결합 (Mean, Std 포함) -> (N_trials, n_channels, 2)
    stats = np.concatenate([mean, std], axis=-1)


    save_dir = os.path.join(save_dir,'pkl')
    os.makedirs(save_dir, exist_ok=True)

    if run_n is not None:
        file_name = f"{subject_id}_run{run_n}"
    else:
        file_name = f"{subject_id}"

    pickle.dump(
        {"X": X_all, "y": y_all, "ch_names": ch_names, "stats": stats},
        open(os.path.join(save_dir, f"{file_name}.pkl"), "wb"),
    )

    # print(f"{file_name}: X={X_all.shape}, y={y_all.shape}, y unique={np.unique(y_all)}")
        
        

def process_and_save_data(subject_ids, run_ns, data_dir, output_dir, drop_chs, ch_order, resample_rate, save_format, save_mode):
    """Processes MNE Epochs from .fif files and saves each trial as a .pkl file.

    This function iterates through a list of subject IDs, reads their corresponding
    .fif files, extracts each trial's signal and label, and saves them into
    individual pickle files in the specified output directory.

    Args:
        subject_list (list): A list of subject identifier strings to process.
        output_dir (str): The path to the directory where the .pkl files will be saved.
    """

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing data for: {os.path.basename(output_dir)}")

    sample_info = read_sample_info(subject_ids, data_dir)
    use_channels_names = sample_info["ch_names"]
    print(f"Using {len(use_channels_names)} channels from sample data.")


    for subject_id in tqdm(subject_ids, desc=f"Processing {os.path.basename(output_dir)}"):
        subject_data_buffer = []  # Subject data buffer
        subject_label_buffer = []

        for run_n in run_ns:
            # Construct the full path to the .fif file.
            path = get_file_paths(subject_id, run_n, data_dir)

            try:
                epochs_stim = read_mne_data(path["stim"])
                epochs_iti = read_mne_data(path["iti"])

                epochs_stim = preprocess_epochs(epochs_stim, resample_rate, drop_chs, ch_order)
                epochs_iti = preprocess_epochs(epochs_iti, resample_rate, drop_chs, ch_order)

                x_stim, y_stim, _ = extract_stim(epochs_stim)
                x_iti, y_iti, _ = extract_iti(epochs_iti)  

                x_all = np.concatenate((x_stim, x_iti), axis=0)
                y_all = np.concatenate((y_stim, y_iti), axis=0)
                # breakpoint()

                if save_mode == 'run':
                    if save_format == 'npy' or save_format == 'npz':
                        export_np(subject_id, run_n, x_all, y_all, use_channels_names, output_dir, file_ext=save_format)
                    elif save_format == 'pkl':
                        export_pkl(subject_id, run_n, x_all, y_all, use_channels_names, output_dir)
            except FileNotFoundError:
                # File not found (just print)
                continue
            except Exception as e:
                print(f"Error processing {subject_id} Run {run_n}: {e}")
                continue

            subject_data_buffer.append(x_all)
            subject_label_buffer.append(y_all)
            save_metadata(epochs_stim.info, subject_id, os.path.join(output_dir, save_format))
        
        if save_mode == 'subject':

            X_subject = np.concatenate(subject_data_buffer, axis=0)
            y_subject = np.concatenate(subject_label_buffer, axis=0)

            if save_format == 'npy' or save_format == 'npz':
                export_np(subject_id, None, X_subject, y_subject, use_channels_names, output_dir, file_ext=save_format)
            elif save_format == 'pkl':
                export_pkl(subject_id, None, X_subject, y_subject, use_channels_names, output_dir)

# %% MAIN
# region
None
# endregion
if __name__ == "__main__":
    

    # data_dir = "/local_raid3/03_user/myyu/hackathon_eeg"
    data_dir = "/local_raid3/03_user/hoian/brlBandit/hackathon_eeg"
    output_root = "/local_raid3/03_user/myyu/EEG_decoder/EEG(500Hz)_COMB"

    subject_ids = os.listdir(data_dir)
    run_ns = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

    np.random.seed(42) # settle random seed for reproducibility
    np.random.shuffle(subject_ids)

    train_subjects = subject_ids[:int(len(subject_ids) * 0.8)]
    test_subjects = subject_ids[int(len(subject_ids) * 0.8):]

    print(f"Total subjects: {len(subject_ids)}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

    process_and_save_data(
        subject_ids=train_subjects, run_ns=run_ns, data_dir=data_dir, 
        output_dir= os.path.join(output_root, "processed_train"), 
        drop_chs=None, ch_order=None, resample_rate=None, save_format="npy", save_mode="subject"
    )
    process_and_save_data(
        subject_ids=test_subjects, run_ns=run_ns, data_dir=data_dir, 
        output_dir= os.path.join(output_root, "processed_test"), 
        drop_chs=None, ch_order=None, resample_rate=None, save_format="npy", save_mode="subject"
    )

    print("Save complete.")
