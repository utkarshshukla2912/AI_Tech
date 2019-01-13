from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
        self.regressor_models = []


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


    def prepareDataForRegressor(self,split = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.clean_data_channels_to_keep, self.clean_data_channels_to_reconstruct, test_size=split, random_state=42)
        X_train, X_test, y_train, y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
        X_train, X_test, y_train, y_test = X_train.reshape(-1,X_train.shape[-1]),X_test.reshape(-1,X_test.shape[-1]),y_train.reshape(-1,y_train.shape[-1]),y_test.reshape(-1,y_test.shape[-1])
        self.data_for_regressor = [X_train,X_test,y_train,y_test]



    def prepareDataForGenerativeInpainting(self):
        image_array = np.array(self.clean_data_all_channels)
        image_array = image_array.reshape(len(image_array),10,100,16)
        for i,recording in enumerate(image_array):
            for j,image in enumerate(recording):
                img = Image.fromarray(image,'L')
                img.save(self.object_path+'generative_inpainting/images/'+str(i)+'_'+str(j)+'.png')



    def runRegressor(self,visualise = True):
        X_train,X_test,y_train,y_test = self.data_for_regressor
        for i in range(y_test.shape[1]):
            regressor = RandomForestRegressor().fit(X_train,y_train[:,i])
            print('Score for predicting channel '+str(regressor.score(X_test,y_test[:,i])*100))
            if visualise:
                y_predicted = regressor.predict(X_test)
                a, = plt.plot(y_test[:,i],color = 'r',label="Original")
                b, = plt.plot(y_predicted,color = 'g',label="Predicted by Random Forest",linestyle='--')
                plt.show()
            self.regressor_models.append(regressor)
        self.save_obj(self.regressor_models,'regressor_models')


pipeline = EEG_pipeline()
pipeline.readRecordings()
pipeline.preProcessData()
pipeline.prepareDataForRegressor()
pipeline.prepareDataForGenerativeInpainting()
#pipeline.runRegressor()
#









#done
