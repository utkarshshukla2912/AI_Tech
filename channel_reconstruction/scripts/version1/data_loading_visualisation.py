import warnings
warnings.filterwarnings('ignore')

from mne.viz import plot_raw,plot_raw_psd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import mne

def save_obj(obj, name ):
    with open('../../objects/version1/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

to_read = ['1_kunhee_0.csv', '1_kunhee_1.csv', '1_kunhee_2.csv', '2_kunhee_0.csv', '2_kunhee_1.csv', '2_kunhee_2.csv', '3_kunhee_0.csv', '3_kunhee_1.csv', '3_kunhee_2.csv', '4_kunhee_0.csv', '4_kunhee_1.csv', '4_kunhee_2.csv', '5_kunhee_0.csv', '5_kunhee_1.csv', '5_kunhee_2.csv', '6_kunhee_0.csv', '6_kunhee_1.csv', '6_kunhee_2.csv', '7_kunhee_0.csv', '7_kunhee_1.csv', '7_kunhee_2.csv', '8_kunhee_0.csv', '8_kunhee_1.csv', '8_kunhee_2.csv', '9_kunhee_0.csv', '9_kunhee_1.csv', '9_kunhee_2.csv', '10_kunhee_0.csv', '10_kunhee_1.csv', '10_kunhee_2.csv', '11_kunhee_0.csv', '11_kunhee_1.csv', '11_kunhee_2.csv', '12_kunhee_0.csv', '12_kunhee_1.csv', '12_kunhee_2.csv', '13_kunhee_0.csv', '13_kunhee_1.csv', '13_kunhee_2.csv', '14_kunhee_0.csv', '14_kunhee_1.csv', '14_kunhee_2.csv', '15_kunhee_0.csv', '15_kunhee_1.csv', '15_kunhee_2.csv', '16_kunhee_0.csv', '16_kunhee_1.csv', '16_kunhee_2.csv']

columns = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']
to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4"]

sfreq = 128
df = None
dfs = []

for file in tqdm(to_read):
    df = pd.read_csv('../../data/rawInput/'+file,header=None)
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

raw = mne.io.RawArray(data, info)
raw.set_channel_types(mapping)
montage = mne.channels.read_montage('standard_1020')
raw.set_montage(montage)
scalings = 'auto'

# Plot Power Spectral Density
#plot_raw_psd(raw)
#plt.show()

save_obj(raw,'Eeg_recordings')
















#done
