from features import featureExtractionPipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json
import glob



with open('config.json') as f:
    config = json.load(f)
data_path = config['data_path']
object_path = config['object_path']
sampling_frequency = config['sampling_frequency']
files = glob.glob(data_path+'*.csv')
label = []
features = []
got_label = False
got_data = False
temp_label = 0


featureType = int(input(' Enter Feature Type 1 for channel based, 2 for file based: '))

if featureType == 1:
    # Channel Based Features
    for file in tqdm(files[:10]):
        try:
            temp_label = int(file.split('_')[2])
            got_label = True
        except Exception as e:
            print(e)
            got_label = False
        try:
            df = pd.read_csv(file,header = None)
            df = df.drop(df.index[len(df)-1])
            channel_feature = []
            for i in tqdm(range(len(df))):
                channel_values = featureExtractionPipeline(df.loc[i][:1000]).runPipeline()
                for j in channel_values:
                    for k in j:
                        channel_feature.append(k)
                if got_label == True:
                    features.append(channel_feature)
                    label.append(temp_label)
                channel_feature = []
        except:
            pass
        got_label = False
        got_data = False


    np.save(object_path+'features_per_channel.npy',features)
    np.save(object_path+'label_per_channel.npy',label)



# File Based Features
else:
    for file in tqdm(files[:10]):

        try:
            temp_label = int(file.split('_')[2])
            got_label = True
        except Exception as e:
            print(e)
            got_label = False

        try:
            df = pd.read_csv(file,header = None)
            df = df.drop(df.index[len(df)-1])
            channel_feature = []
            for i in tqdm(range(len(df))):
                channel_values = featureExtractionPipeline(df.loc[i][:1000]).runPipeline()
                for j in channel_values:
                    for k in j:
                        channel_feature.append(k)
            got_data = True
        except Exception as e:
            print(e)
            got_data = False

        if got_label and got_data:
            features.append(np.array(channel_feature))
            label.append(temp_label)
        got_label = False
        got_data = False

    np.save(object_path+'features_file.npy',features)
    np.save(object_path+'label_file.npy',label)







#done
