import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pickle
import mne


def load_obj(name):
    with open('../../objects/version2/'+ name + '.pkl', 'rb') as f:
        object = pickle.load(f)
    f.close()
    return object

raw = load_obj('Eeg_recordings')
numpy_array = raw.get_data()
X = numpy_array[:10].transpose()
Y = numpy_array[10:]

for y in Y:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = RandomForestRegressor().fit(X_train,y_train)
    print('Score for predicting channel '+str(regressor.score(X_test,y_test)*100))
