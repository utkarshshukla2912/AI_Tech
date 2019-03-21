from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import json
import os


def getScaledData(df):
	labels = df.loc[len(df)-1].values
	df = df.drop(df.index[len(df)-1])
	df = list(StandardScaler().fit_transform(np.array(df)))
	df.append(labels)
	df = pd.DataFrame(df)
	return(df)

def getDiffData(df):
	labels = df.loc[len(df)-1].values[:-1]
	df = df.drop(df.index[len(df)-1])
	channelList = np.array(df)
	temp = []
	for channel in channelList:
		temp.append([t - s for s, t in zip(channel, channel[1:])])
	temp.append(labels)
	df = pd.DataFrame(np.array(temp))
	return(df)

with open('config.json') as f:
	config = json.load(f)

files = glob.glob(config['dataPath']+'*.csv')
for file in tqdm(files):
	df = pd.read_csv(file,header = None)
	break
