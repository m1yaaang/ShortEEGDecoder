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


CHANNEL_DICT = {k.upper():v for v,k in enumerate(     #62 ch
                    [      'FP1', 'FPZ', 'FP2', 
                    "AF7", 'AF3', 'AF4', "AF8", 
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                    'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                            'O1', 'OZ', 'O2', ])}


drop_channels = ['TP9', 'TP10', 'FT9', 'FT10']
chOrder_standard = [          # 57-4 = 53ch
                            'Fpz',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                    'O1', 'O2', 'Oz']


# def read_fif(data_dir):
# read data
data_dir = "/local_raid3/03_user/myyu/hackathon_eeg"
output_root = "/local_raid3/03_user/myyu/EEGPT/downstream_combine3/PreprocessedEEG/"

# subject_ids = ["16", "17", "22", "27", "33", "39", "40", "56", "58", "59", "64", "68", "69", "72", "79"]
subject_ids = os.listdir(data_dir)
run_ns = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]



# CHANNEL_DICT = {k.upper():v for v,k in enumerate(     #62 ch
#                  [      'FP1', 'FPZ', 'FP2', 
#                     "AF7", 'AF3', 'AF4', "AF8", 
#         'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
#         'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
#     'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#          'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
#                   'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
#                            'O1', 'OZ', 'O2', ])}



# null_file_path = data_dir / f"sub-{subject_id}" / "TRI" / "epochs_data" / f"sub-{subject_id}_run{run_n}_iti_icaRej-epo.fif"
# read null data

def process_and_save_data(subject_list, output_folder):
    """Processes MNE Epochs from .fif files and saves each trial as a .pkl file.

    This function iterates through a list of subject IDs, reads their corresponding
    .fif files, extracts each trial's signal and label, and saves them into
    individual pickle files in the specified output directory.

    Args:
        subject_list (list): A list of subject identifier strings to process.
        output_folder (str): The path to the directory where the .pkl files will be saved.
    """


    # Create the output directory if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nProcessing data for: {os.path.basename(output_folder)}")
    


    for subject_id in tqdm(subject_list, desc=f"Processing {os.path.basename(output_folder)}"):
        for run_n in run_ns:
            # Construct the full path to the .fif file.
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

            # Skip if the file does not exist.
            if not os.path.exists(stim_file_path) or not os.path.exists(iti_file_path) :
                continue

            try:
                # Read the MNE Epochs file.
                #####################################################
                #               stim -> not null
                #####################################################
                epochs = mne.read_epochs(stim_file_path, preload=True, verbose=False, )
                epochs.resample(sfreq=256)
        
                info = epochs.info['ch_names']
                # assert epochs.info['ch_names'] == use_channels_names, f"channel order is different from channel_ref"
                # print(info)
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in info:
                            useless_chs.append(ch)
                    epochs.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(epochs.info['ch_names']):
                    epochs.reorder_channels(chOrder_standard)
                # Iterate through each trial within the Epochs object.

                # Extract data and event labels into NumPy arrays.
                X = epochs.get_data()      # Shape: (n_trials, n_channels, n_times)
                y = epochs.events[:, 2]    # Shape: (n_trials,)
                
                # print(X.shape)
                for i in range(len(X)):
                    single_trial_data = X[i]
                    single_trial_label = y[i]
                    
                    # Create a dictionary to store the signal and its corresponding label.
                    sample = {
                        "signal": single_trial_data, 
                        "label": single_trial_label,
                        "ch_names": info
                    }
                    
                    # Generate a unique filename for each trial.
                    output_filename = f"{subject_id}_run-{run_n}_trial-{i:03d}.pkl"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save the dictionary as a pickle file.
                    with open(output_path, 'wb') as f:
                        pickle.dump(sample, f)

                #####################################################
                #               iti -> null
                #####################################################
                epochs = mne.read_epochs(iti_file_path, preload=True, verbose=False, )
                epochs.resample(sfreq=256)
        
                info = epochs.info['ch_names']
                # assert epochs.info['ch_names'] == use_channels_names, f"channel order is different from channel_ref"
                # print(info)
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in info:
                            useless_chs.append(ch)
                    epochs.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(epochs.info['ch_names']):
                    epochs.reorder_channels(chOrder_standard)
                # Iterate through each trial within the Epochs object.

                # Extract data and event labels into NumPy arrays.
                X = epochs.get_data()      # Shape: (n_trials, n_channels, n_times)
                y = epochs.events[:, 2]    # Shape: (n_trials,)
                
                # print(X.shape)
                n_trials = X.shape[0]
                n_keep = int(n_trials * 0.2)  # 1/5
                selected_idx = np.random.choice(n_trials, size=n_keep, replace=False)
                for i in selected_idx:
                    single_trial_data = X[i]
                    # null data should be 1
                    # interval time(1s) between objects 
                    single_trial_label = 1
                    
                    # Create a dictionary to store the signal and its corresponding label.
                    sample = {
                        "signal": single_trial_data, 
                        "label": single_trial_label,
                        "ch_names": info
                    }
                    
                    # Generate a unique filename for each trial.
                    output_filename = f"{subject_id}_run-{run_n}_trial-{i:03d}-null.pkl"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save the dictionary as a pickle file.
                    with open(output_path, 'wb') as f:
                        pickle.dump(sample, f)

                

            except Exception as e:
                print(f"Error processing file {stim_file_path}: {e}")

def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


# %% MAIN
# region
None
# endregion
if __name__ == "__main__":
    
    # np.random.seed(42) # settle 
    # np.random.shuffle(subject_ids)

    # train_subjects = subject_ids[:int(len(subject_ids) * 0.8)]
    # # val_subjects = subject_ids[int(len(subject_ids) * 0.7):int(len(subject_ids) * 0.8)]
    # test_subjects = subject_ids[int(len(subject_ids) * 0.8):]

    # print(f"Total subjects: {len(subject_ids)}")
    # print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    # # print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
    # print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

    # process_and_save_data(train_subjects, os.path.join(output_root, "processed_train"))
    # # process_and_save_data(val_subjects, os.path.join(output_root, "processed_eval"))
    # process_and_save_data(test_subjects, os.path.join(output_root, "processed_test"))
    


    print(f"Total subjects: {len(subject_ids)}")

    process_and_save_data(subject_ids, os.path.join(output_root, "processed_All"))
