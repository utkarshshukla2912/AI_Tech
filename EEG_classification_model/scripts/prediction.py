from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import SeparableConv2D, DepthwiseConv2D
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras.regularizers import l1_l2
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import json
import glob
import os

K.set_image_dim_ordering('th')
logging.basicConfig(filename = 'logs/prediction.logs',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_file(file):
    recording = pd.read_csv(file,header = None)
    recording = recording.drop(recording.index[len(recording)-1])
    recording = np.array(recording).reshape(-1,1).[:,:1000]
    recording = recording.reshape(len(recording),1, 16, 1000)
    return(recording)

def getPredictions(data_path,object_path,model_name,nb_classes):
    model = load_model(object_path+model_name+'.h5')
    prediction = model.predict(recording, batch_size=32)
    print(np.argmax(prediction))


def predictionPipeline():
    logger.info('Starting Prediction Pipeline')
    with open('config.json') as f:
        config = json.load(f)
    model_name = config['training_model']
    data_path = config['data_path']
    object_path = config['object_path']
    nb_classes = config['nb_classes']
    getPredictions(data_path,object_path,model_name,nb_classes)
