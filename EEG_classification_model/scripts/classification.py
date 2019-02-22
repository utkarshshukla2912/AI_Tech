from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import SeparableConv2D, DepthwiseConv2D
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import glob
import os

K.set_image_dim_ordering('th')
logging.basicConfig(filename = 'model_logs.logs',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def EEGNet(nb_classes, Chans = 16, Samples = 100,dropoutRate = 0.25,
           kernLength = 64, F1 = 4,D = 2, F2 = 8, norm_rate = 0.25,
            dropoutType = 'Dropout'):

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

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))

def ShallowConvNet(nb_classes, Chans = 16, Samples = 100, dropoutRate = 0.5):

    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(40, (1, 13),
                                 input_shape=(1, Chans, Samples),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False,
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)

def loadData(location = '../data/*'):

    if not os.path.exists('../objects/data_collection.npy') or not os.path.exists('../objects/label_collection.npy'):
        logging.info('Fetching Data from raw files')
        data_collection = []
        label_collection = []
        file_list = glob.glob(location)
        for file in tqdm(file_list):
            recording = pd.read_csv(file,header = None)
            recording = recording.drop(recording.index[len(recording)-1])
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

def dataPreprocessor():

    data = loadData()
    enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
    eeg_data = data[0]
    labels = data[1]
    resampled_data = []
    resampled_labels = []
    for i in range(len(eeg_data)):
        label = labels[i]
        for j in range(10):
            ed = eeg_data[i][:,j*100:(j+1)*100]
            resampled_data.append(ed)
            resampled_labels.append(label)
    resampled_data = np.array(resampled_data)
    resampled_data = resampled_data.reshape(len(resampled_data),1, 16, 100)
    resampled_labels = np.array(resampled_labels)
    resampled_labels = resampled_labels.reshape(-1,1)
    resampled_labels = np.array(enc.fit_transform(resampled_labels),dtype = int)
    return([resampled_data,resampled_labels])

def DeepConvNet(nb_classes, Chans = 16, Samples = 100,
                dropoutRate = 0.5):

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(25, (1, 5),
                                 input_shape=(1, Chans, Samples),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)

    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)

    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)

    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)

    flatten      = Flatten()(block4)

    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def trainModel():

    data = dataPreprocessor()
    #model = EEGNet(nb_classes = 2)
    #model = ShallowConvNet(nb_classes = 2)
    model = DeepConvNet(nb_classes = 2)
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit(data[0], data[1], batch_size=128, epochs=250,validation_split=0.2)
    return(history)

def modelVis():
    history = trainModel()
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../objects/accuracy.png')
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../objects/loss.png')




modelVis()


#done
