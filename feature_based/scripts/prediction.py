from features import featureExtractionPipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json
import glob


file_path = "../../data/1_andreas_1_20181222-223846.csv" # enter file path

with open('config.json') as f:
    config = json.load(f)

object_path = config['object_path']
model_to_use = config['model_to_use']
models_file_based = config['models_file_based']
models_channel_based = config['models_channel_based']

df = pd.read_csv(file_path,header = None)
df = df.drop(df.index[len(df)-1])

if model_to_use == 'file_based':
    channel_feature = []
    for i in tqdm(range(len(df))):
        channel_values = featureExtractionPipeline(df.loc[i][:1000]).runPipeline()
        for j in channel_values:
            for k in j:
                channel_feature.append(k)

    models = glob.glob(object_path + models_file_based + '*.sav')
    for m in models:
        with open(m, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
            print(m[m.rindex('/')+1:],' -> ',model.predict([channel_feature])[0])

else:
    channel_feature = []
    for i in tqdm(range(len(df))):
        channel_values = featureExtractionPipeline(df.loc[i][:1000]).runPipeline()
        for j in channel_values:
            for k in j:
                channel_feature.append(k)
        models = glob.glob(object_path + models_channel_based + '*.sav')
        for m in models:
            with open(m, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
                print(m[m.rindex('/')+1:],' -> ',model.predict([channel_feature]))
        channel_feature = []

#done
