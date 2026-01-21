filepath = "/home/winter/hoian/brlBandit/prep-obs_pc1/"

import mne

raw = mne.io.read_raw_eeglab(set_files, preload=True)

# 1) Filtering
raw.filter(l_freq = 1.0, h_freq = 40.0)

# 2) Notch(60Hz)
raw.notch_filter(freqs=[60])

# 3) average reference
raw.set_eeg_reference('average')

# 4) ICA
ica = mne.preprocessing.ICA(n_components=0.99, random_state=0,max_iter="auto")
ica.fit(raw)

raw_clean = ica.apply(raw.copy())

raw_clean.save(f"sub01_clean_raw.fif")