from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob

to_read = ['1_kunhee_0.csv', '1_kunhee_1.csv', '1_kunhee_2.csv', '2_kunhee_0.csv', '2_kunhee_1.csv', '2_kunhee_2.csv', '3_kunhee_0.csv', '3_kunhee_1.csv', '3_kunhee_2.csv', '4_kunhee_0.csv', '4_kunhee_1.csv', '4_kunhee_2.csv', '5_kunhee_0.csv', '5_kunhee_1.csv', '5_kunhee_2.csv', '6_kunhee_0.csv', '6_kunhee_1.csv', '6_kunhee_2.csv', '7_kunhee_0.csv', '7_kunhee_1.csv', '7_kunhee_2.csv', '8_kunhee_0.csv', '8_kunhee_1.csv', '8_kunhee_2.csv', '9_kunhee_0.csv', '9_kunhee_1.csv', '9_kunhee_2.csv', '10_kunhee_0.csv', '10_kunhee_1.csv', '10_kunhee_2.csv', '11_kunhee_0.csv', '11_kunhee_1.csv', '11_kunhee_2.csv', '12_kunhee_0.csv', '12_kunhee_1.csv', '12_kunhee_2.csv', '13_kunhee_0.csv', '13_kunhee_1.csv', '13_kunhee_2.csv', '14_kunhee_0.csv', '14_kunhee_1.csv', '14_kunhee_2.csv', '15_kunhee_0.csv', '15_kunhee_1.csv', '15_kunhee_2.csv', '16_kunhee_0.csv', '16_kunhee_1.csv', '16_kunhee_2.csv']
sensor_name = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']
to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4"]
sfreq = 128
df = None
dfs = []
for file in tqdm(to_read):
    df = pd.read_csv('../../data/rawInput/'+file,header=None)
    df = df.transpose()
    df.columns = sensor_name
    dfs.append(df)

combined = pd.concat(dfs)
to_hide.append('Label')
X = combined.drop(to_hide,axis = 1)
Y = combined['F3']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
l = RandomForestRegressor().fit(X_train,y_train)

p = l.predict(X_test)
for i in range(100):
    print(p[i],list(y_test)[i])




    '''
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import numpy as np
    import pickle
    import mne

    def load_obj(name):
        with open('../../objects/version1/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_obj(obj, name ):
        with open('../../objects/version1/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    raw = load_obj('Eeg_recordings')
    eeg_recordings = raw.get_data().tolist()
    channels = raw.info['ch_names']
    channels_to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4"]
    hidden_channel_value = []
    channel_models = {}

    for i in channels_to_hide:
        index = channels.index(i)
        hidden_channel_value.append(eeg_recordings[index])
        eeg_recordings.pop(index)

    for channel in hidden_channel_value:
        X = eeg_recordings
        Y = channel
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        regressor = RandomForestRegressor().fit(X_train,y_train)
        print('Regressor performance for predicting channel '+channel,regressor.score(X_test,y_test)*100)
        channel_models[channel] = regressor

    save_obj(channel_models,'channel_models')

    '''
