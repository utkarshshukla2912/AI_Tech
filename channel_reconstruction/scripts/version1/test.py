import warnings
warnings.filterwarnings('ignore')

from mne.viz import plot_raw,plot_raw_psd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import glob
import mne

def save_obj(obj, name ):
    with open('../../objects/version1/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



columns = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']
to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4"]

sfreq = 128
df = None
dfs = []

for file in tqdm(glob.glob('../../data/version2/*.csv')[:20]):
    df = pd.read_csv(file,header=None)
    df = df.transpose()
    df.columns = columns
    dfs.append(df)

combined = pd.concat(dfs)
del dfs
del df

ch_names = columns[:len(columns)-1]
info = mne.create_info(ch_names, sfreq)
data = []
mapping = {}
for i in ch_names:
    data.append(list(combined[i]))
    mapping[i] = 'eeg'

raw = mne.io.RawArray(data, info,verbose=None)
raw.set_channel_types(mapping)
montage = mne.channels.read_montage('standard_1020')
raw.set_montage(montage,verbose=None)
scalings = 'auto'
raw.plot(scalings = scalings)
plot_raw_psd(raw)
plt.show()

save_obj(raw,'Eeg_recordings')
















#done
