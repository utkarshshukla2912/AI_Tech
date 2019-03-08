# Python stuff
import os
import signal
import numpy as np
import datetime
from itertools import cycle
import random
import time
from subprocess import call, Popen, PIPE
import csv

# Written Classes
from showSlides import ScreenView
from dataOrganize import Data_Organize
from model_colors.model_initial import Data_prep, EEGNet, Evaluation

from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

#VERY IMPORTANT FOR CONSISTENCY seed(0)
torch.manual_seed(0)

SIZE_NET= 16*62
COLORS = 2

# options: red 1, blue 2, black 3, cian 4, orange 5,
colourDict  = {}
path  = '/home/andrea/Bureau/aitech/projects/bci/code/bci/experiment/'
if COLORS == 5:
    colourDict = {
        "red": 1,
        "blue": 2,
        "black": 3,
        "cian": 4,
        "orange": 5,
    }
elif  COLORS==2:
    colourDict = {
        "red": 1,
        "blue": 2,
    }
else:
    raise ("Error no matching colors dict")


colour_seq = random.sample(colourDict.keys(),1)
image_colour_seq = []
image_label_seq = []
# [image_files.extend(['white.gif','dot.gif',imageName + '.gif']) for imageName in colour_seq]
for imageName in colour_seq:
    image_colour_seq.extend(['white.gif', 'dot.gif', imageName + '.gif'])
    image_label_seq.extend([colourDict[imageName]])

"""
====================================================================================================================
Run eval
====================================================================================================================
"""
print('Run evaluation')
file_dir = '/home/andrea/Bureau/aitech/projects/bci/code/bci/experiment/'
# Prepare data for model training
prep= Data_prep()
fileName = '1_testonline_2_20190207-211044.csv'#'1_testonline_1_20190202-163051.csv'#'testblue.csv'#'1_testonline_2_20190202-162446.csv'####'##''2colours/3_eliseo_2_20190104-190905.csv'

print('file to test: ', fileName)


# kun_1, kun_2 = prep.csv2arrays(fstart=100,fend=101,name='kunhee', file_dir=file_dir, data_type=[1,2], files=files)
kun_1 = np.genfromtxt(file_dir+fileName, delimiter=',').astype('float32')[:-1,:]
kun_1 = kun_1[:,:SIZE_NET]

X , y = prep.list_2darrays_to_3d([kun_1],[-1])

# Load pre-trained model
net = EEGNet(SIZE_NET)
load_path = file_dir+'/model_colors/eeg_net_20190209.pt'

net.load_state_dict(torch.load(load_path))
net.eval()




array_dstack = np.array(X)
print(X.shape)


# (#samples, 1, #timepoints, #channels)
array_dstack_reshaped = np.reshape(array_dstack,(1, 1, SIZE_NET, 16))
inputs = Variable(torch.from_numpy(array_dstack_reshaped))
pred = net(inputs)

print(pred)




indx = int(np.argmax(pred.detach().numpy()))
#print (colourDict.values(),'--',indx)
color =  tuple(colourDict.items())[indx][0]
realcolor = tuple(colourDict.items())[image_label_seq[0]-1][0]
print ('Color detected: ',color)


print('done')
"""
====================================================================================================================
"""
