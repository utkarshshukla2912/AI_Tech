from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import SeparableConv2D, DepthwiseConv2D
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras.regularizers import l1_l2
from keras import backend as K
from keras.models import Model
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import glob
import os

K.set_image_dim_ordering('th')

logging.basicConfig(filename = 'eeg_net.logs',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def EEGNet(nb_classes, Chans = 16, Samples = 1000,
             dropoutRate = 0.25, kernLength = 64, F1 = 4,
             D = 2, F2 = 8, norm_rate = 0.25, dropoutType = 'Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (1, Chans, Samples))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
    flatten      = Flatten(name = 'flatten')(block2)
    dense        = Dense(nb_classes, name = 'dense',
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    return Model(inputs=input1, outputs=softmax)


def loadData(location = '../data/*'):

    if not os.path.exists('../objects/data_collection.npy') or not os.path.exists('../objects/label_collection.npy'):
        logging.info('Fetching Data from raw files')
        data_collection = []
        label_collection = []
        file_list = glob.glob(location)
        for file in tqdm(file_list):
            recording = pd.read_csv(file)
            recording = np.array(recording)[:,:1000]
            data_collection.append(recording)
            label_collection.append(int(float(file.split('_')[2])))
        logging.info('Saving Data Fetched From Raw Files')
        np.save('../objects/data_collection.npy',data_collection)
        logging.info('Saving Labels Fetched From Raw Files')
        np.save('../objects/label_collection.npy',label_collection)
        return [data_collection,label_collection]
    else:
        logging.info('Data and Labels exists')
        data_collection = np.load('../objects/data_collection.npy')
        label_collection = np.load('../objects/label_collection.npy')
        logging.info('Existing Data and Labels fetched')
        return [data_collection,label_collection]

model = EEGNet(nb_classes = 2)
model.summary()
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
data = loadData()
eeg_data = data[0]
resampled_data = []

eeg_data = eeg_data.reshape(len(eeg_data),1, 16, 1000)

labels = np.array(data[1]).reshape(-1,1)
labels = np.array(enc.fit_transform(labels),dtype = int)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(eeg_data.shape)
model.fit(eeg_data, labels, batch_size=16, nb_epoch=150)







#done
