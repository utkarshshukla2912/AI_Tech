from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import glob
import os


class EEG_pipeline:

    def __init__(self, location = '../data/Eeg_recordings'):

        self.columns = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']
        self.columns_to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4"]
        self.columns_to_keep = ['P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4']
        self.location = location
        self.object_path = '../objects/'
        self.raw_data = []
        self.sampling_freq = 128
        plt.style.use('ggplot')

        self.clean_data_all_channels = []
        self.clean_data_channels_to_reconstruct = []
        self.clean_data_channels_to_keep = []
        self.labels = []
        self.data_for_regressor = []
        self.regressor_models = {'gbr':GradientBoostingRegressor(), 'rfr':RandomForestRegressor(), 'enr':ElasticNet(),'lassor':Lasso()}


    def save_obj(self, obj, name):
        with open(self.object_path+'basic_approach/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()


    def load_obj(self, name):
        with open(self.object_path+'basic_approach/'+ name + '.pkl', 'rb') as f:
            object = pickle.load(f)
        f.close()
        return object


    def readRecordings(self):
        if os.path.exists(self.object_path+'basic_approach/raw_data.pkl'):
            self.raw_data = self.load_obj('raw_data')
        else:
            recordings = glob.glob(self.location+'/*.csv')
            recordings.sort()
            for recording in tqdm(recordings):
                df = pd.read_csv(recording,header=None)
                df = df.T.astype(float)
                df.columns = self.columns
                self.raw_data.append(df)
            self.save_obj(self.raw_data,'raw_data')


    def preProcessData(self):
        if os.path.exists(self.object_path+'basic_approach/clean_data_all_channels.pkl') and os.path.exists(self.object_path+'basic_approach/clean_data_channels_to_keep.pkl') and os.path.exists(self.object_path+'basic_approach/clean_data_channels_to_reconstruct.pkl') and os.path.exists(self.object_path+'basic_approach/labels.pkl'):
            self.clean_data_all_channels = self.load_obj('clean_data_all_channels')
            self.clean_data_channels_to_reconstruct = self.load_obj('clean_data_channels_to_keep')
            self.clean_data_channels_to_keep = self.load_obj('clean_data_channels_to_reconstruct')
            self.labels = self.load_obj('labels')
        else:
            for data in tqdm(self.raw_data):
                self.clean_data_all_channels.append(data.drop('Label',axis = 1).values[:1000])
                self.clean_data_channels_to_keep.append(data[self.columns_to_keep].values[:1000])
                self.clean_data_channels_to_reconstruct.append(data[self.columns_to_hide].values[:1000])
                self.labels.append(int(data['Label'][0]))

        self.save_obj(self.clean_data_all_channels,'clean_data_all_channels')
        self.save_obj(self.clean_data_channels_to_reconstruct,'clean_data_channels_to_reconstruct')
        self.save_obj(self.clean_data_channels_to_keep,'clean_data_channels_to_keep')
        self.save_obj(self.labels,'labels')


    def runRegressor(self,visualise = False):
        for regressor in self.regressor_models:
            print('Training Model:',regressor)
            model_array = []
            for _ in range(0,6):
                model_array.append(self.regressor_models[regressor])
            kf = KFold(n_splits=10)
            splits = kf.split(self.clean_data_channels_to_keep)
            count = 1
            for train_index, test_index in splits:
                print('\tRunning split number: ',count); count += 1
                X_train = np.take(self.clean_data_channels_to_keep,train_index,axis = 0)
                X_test = np.take(self.clean_data_channels_to_keep,test_index,axis = 0)
                y_train = np.take(self.clean_data_channels_to_reconstruct,train_index,axis = 0)
                y_test = np.take(self.clean_data_channels_to_reconstruct,test_index,axis = 0)
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 10)
                X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], 10)
                y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], 6)
                y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], 6)
                for i in range(y_train.shape[1]):
                    #print('\tTraining for Channel:',self.columns_to_hide[i])
                    X_train = StandardScaler().fit_transform(X_train)
                    model_array[i].fit(X_train,y_train[:,i])
                    #print('\t\tScore for predicting channel '+str(self.regressor_models[i].score(StandardScaler().fit_transform(X_test),y_test[:,i])*100))
                    if visualise:
                        y_predicted = regressor.predict(X_test)
                        a, = plt.plot(y_test[:,i],color = 'r',label="Original")
                        b, = plt.plot(y_predicted,color = 'g',label="Predicted by Random Forest",linestyle='--')
                        plt.show()
            self.save_obj(model_array,'regressor_models_'+regressor)

    def prepareDataForGenerativeInpainting(self):
        image_array = np.array(self.clean_data_all_channels)
        image_array = image_array.reshape(len(image_array),10,100,16)
        for i,recording in enumerate(image_array):
            for j,image in enumerate(recording):
                img = Image.fromarray(image,'L')
                img.save(self.object_path+'generative_inpainting/images/'+str(i)+'_'+str(j)+'.png')


pipeline = EEG_pipeline()
pipeline.readRecordings()
pipeline.preProcessData()
pipeline.prepareDataForGenerativeInpainting()
pipeline.runRegressor() # set visualise = True to see difference between predicted and actual values










#done
