from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import glob
import os


class Reconstruction_pipeline:

    def __init__(self):
        self.model_list = None
        self.object_path = '../objects/'
        self.recordings_path = '../../data/*.csv'
        self.columns_to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4", "FC5", "FC4", "CP5", "CP4"]
        self.columns_to_keep = ['P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
        self.columns = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']

    def save_obj(self, obj, name):
        with open(self.object_path+'random_forest/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_obj(self, name):
        with open(self.object_path+'random_forest/'+ name + '.pkl', 'rb') as f:
            object = pickle.load(f)
        f.close()
        return object

    def getReconstructedRecordings(self):
        self.model_list = self.load_obj('regressor_models')
        files = glob.glob(self.recordings_path)
        for file in tqdm(files):
            df = pd.read_csv(file,header=None)
            df = df.T.astype(float)
            df.columns = self.columns
            input_to_model = df[self.columns_to_keep].values
            input_to_model = input_to_model.reshape(len(input_to_model),6)
            #input_to_model = StandardScaler().fit_transform(input_to_model)
            original_values = df[self.columns_to_hide].values
            for i,col in enumerate(self.columns_to_hide):
                predicted_columns_values = self.model_list[i].predict(input_to_model)
                #print(self.model_list[i].score(input_to_model,df[self.columns_to_hide[i]]))
                df[col] = predicted_columns_values
            file_name = file[file.rindex('/')+1:].replace('.csv','_reconstructed.csv')
            df.to_csv('../objects/reconstructed_files/'+file_name)



pipeline = Reconstruction_pipeline()
pipeline.getReconstructedRecordings()









#done
