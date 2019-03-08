# coding: utf-8


"""
Written by,
Daniel Usvyat

Using:
Sriram Ravindran, sriram@ucsd.edu

Original paper - https://arxiv.org/abs/1611.08024

Please reach out to me if you spot an error.
"""



import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
import time
from datetime import date

#VERY IMPORTANT FOR CONSISTENCY seed(0)
torch.manual_seed(0)

'''
=======================PARAM
'''
SIZE_NET= 16*62


class Data_prep():

    def csv2arrays_old(self,fstart=1,fend=40,
                   file_dir='/home/andrea/Bureau/aitech/projects/bci/dataset/rawInput/',
                   data_type=['1', '2'],
                   name='kunhee'):

        arrays_1=[]
        arrays_2=[]

        for j in data_type:
            for i in range(fstart,fend):
                if '{number}_{name}_{dtype}.csv'.format(number=i,name=name,dtype=j) in files:
                    if j == "1":
                        arrays_1.append(np.genfromtxt(file_dir+'{number}_{name}_{dtype}.csv'.format(number=i,name=name,dtype=j), delimiter=',').astype('float32')[:-1,:])
                    elif j == "2":
                        arrays_2.append(np.genfromtxt(file_dir+'{number}_{name}_{dtype}.csv'.format(number=i,name=name,dtype=j), delimiter=',').astype('float32')[:-1,:])
                else:
                    continue
        return arrays_1, arrays_2

    def csv2arrays(self,
                   file_dir,cut_size,
                   data_type=['1', '2'],
                   min_size=1000,
                   invert=False):
        cntred = 0
        cntblue = 0
        arrays_1 = []
        files1 = []
        files2 = []
        arrays_2 = []
        min_size_dataset = min_size
        for f in files:
            #print (f.split('_'))
            expNo, name, col, *_ = f.split('_')
            arr = np.genfromtxt(file_dir + f, delimiter=',').astype('float32')[:-1, :]
            if invert:
                print ('invert')
                arr = np.swapaxes(arr,1,0)
            arr = arr[:, :cut_size]
            print  (arr.shape)
            if col == '1':
                arrays_1.append(arr)
                files1.append(f)
                cntred +=1
            elif col == '2':
                arrays_2.append(arr)
                cntblue +=1
                files2.append(f)

        print ('red:',cntred,'  cntblue:',cntblue)


        return arrays_1, arrays_2, files1, files2


    def list_2darrays_to_3d(self,array,label):
        array_dstack = np.dstack(array)

        #(#samples, 1, #timepoints, #channels)
        array_dstack_reshaped = np.reshape(array_dstack,(array_dstack.shape[2],1,array_dstack.shape[1],array_dstack.shape[0]))
        y = np.full(array_dstack_reshaped.shape[0], label,dtype=np.int64)
        print ('list_2darrays_to_3d: ',label,'--',y)
        return array_dstack_reshaped, y


    def get_one_hot(self,targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes]).astype(np.int64)


class EEGNet(nn.Module):
    def __init__(self,size_net):
        super(EEGNet, self).__init__()
        print('Initialize net with size: ',size_net)
        self.T = size_net

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1,16), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.

        self.fc1 = nn.Linear(int(self.T/2), 2)



    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        #print "layer 1"
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        #print "layer 2"

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        #print "layer 3"

        # FC Layer
        #print ('view:',x.shape)
        x = x.view(-1, int(self.T/2))
        #x = torch.sigmoid(self.fc1(x))
        x= torch.softmax(self.fc1(x),1)


        #print "layer 4"

        return x



# #### Evaluate function returns values of different criteria like accuracy, precision etc.
# In case you face memory overflow issues, use batch size to control how many samples get evaluated at one time. Use a batch_size that is a factor of length of samples. This ensures that you won't miss any samples.

class Evaluation(Data_prep):

    def evaluate(self,model, X_test, y_test, fileslist, params = ["acc"], verbose=False,check=False):
        results = []
        batch_size = 100

        predicted = []
        '''
        for i in range(int(len(X)/batch_size)):
            s = i*batch_size
            e = i*batch_size+batch_size

            inputs = Variable(torch.from_numpy(X[s:e]))
            pred = model(inputs)

            predicted.append(pred.data.cpu().numpy())
        '''

        inputs = Variable(torch.from_numpy(X_test))
        predicted = model(inputs)


        predicted = predicted.data.cpu().numpy()
        print('predicted shape: ',predicted.shape)
        print('shape...',len(y_test),'--',len(fileslist))

        if verbose:
            for i in range(len(predicted)):
                print('preds: ', predicted[i], '   label:', y_test[i],'  file: ',fileslist[i])


        if check:
            diff = []
            print('Num tests: ',len(predicted))
            for i in range(len(predicted)):
                d = int(np.argmax(predicted[i])- np.argmax(y_test[i]))
                if d != 0:
                    print('ERROR preds: ', predicted[i], '   label:', y_test[i], 'file: ',fileslist[i])

                elif np.sum(predicted[i]) < 0.5:
                    print('preds no near 1-sum: ', predicted[i], '   label:', y_test[i], 'file: ',fileslist[i])
                else:
                    print('file sum1: ',fileslist[i])

                diff.append(d)

            toterr = sum(diff)/float(len(diff))
            print('error: ',toterr)

        for param in params:
            if param == 'acc':
                results.append(accuracy_score(y_test, np.round(predicted)))
            if param == "auc":
                results.append(roc_auc_score(y_test, predicted))
            if param == "recall":
                results.append(recall_score(y_test, np.round(predicted), average=None))
            if param == "precision":
                results.append(precision_score(y_test, np.round(predicted), average=None))
            if param == "fmeasure":
                precision = precision_score(y_test, np.round(predicted), average=None)
                recall = recall_score(y_test, np.round(predicted), average=None)
                results.append(2*precision*recall/ (precision+recall))
        return results


    def train(self,n_epochs=30, test_perc = 0.1, n_classes=2):

        n_indices = len(X)
        test_indices = int(n_indices * test_perc)
        train_indices = n_indices - test_indices

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            print ("\nEpoch ", epoch+1)


            running_loss = 0.0

            inputs = torch.from_numpy(X_train)
            labels = torch.FloatTensor(np.array(self.get_one_hot(y_train,n_classes)))


            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()



            optimizer.step()

            running_loss += loss.item()#data[0]


            #print('train len:',X_train.shape[0],'  test len: ',X_test.shape[0])
            # Validation accuracy
            params = ["acc", "auc", "fmeasure"]
            print (params)
            print ("Training Loss ", running_loss)
            print ("Train - ", self.evaluate(net, X_train, self.get_one_hot(y_train,n_classes), params))
            #print "Validation - ", evaluate(net, X_val, y_val, params)
            print ("Test - ", self.evaluate(net, X_test,self.get_one_hot(y_test,n_classes), params))
    # #### Stratified KFold Validation


    def train_Kfold_validation(self,n_epochs=50,n_splits=3,n_classes=2):

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            print ("\nEpoch ", epoch+1)

            k_fold = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=1)

            for train_indices, test_indices in k_fold.split(X,y):



                running_loss = 0.0



                inputs = torch.from_numpy(X[train_indices])
                labels = torch.FloatTensor(np.array(self.get_one_hot(y[train_indices],n_classes)))


                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                #print('c: ',inputs.shape)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()



                optimizer.step()

                running_loss += loss.item()#.data[0]

                # Validation accuracy
                params = ["acc", "auc", "fmeasure"]
                print (params)
                print ("Training Loss ", running_loss)
                print ("Train - ", self.evaluate(net, X[train_indices], self.get_one_hot(y[train_indices],n_classes), params) )
                #print "Validation - ", evaluate(net, X_val, y_val, params)
                print ("Test - ", self.evaluate(net, X[test_indices],self.get_one_hot(y[test_indices],n_classes), params))


    # #### Batch_training no CV

    def train_mini_batch(self,batch_size=16,n_epochs=30,n_classes=2):
        batch_size = batch_size

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            print ("\nEpoch ", epoch)

            running_loss = 0.0
            for i in range(len(X_train)/batch_size-1):
                s = i*batch_size
                e = i*batch_size+batch_size

                inputs = torch.from_numpy(X_train[s:e])
                labels = torch.FloatTensor(np.array(self.get_one_hot(y_train[s:e],2)))

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()


                optimizer.step()

                running_loss += loss.data[0]

            # Validation accuracy
            params = ["acc", "auc", "fmeasure"]
            print (params)
            print ("Training Loss ", running_loss)
            print ("Train - ", self.evaluate(net, X_train, self.get_one_hot(y_train,2), params))
            print ("Validation - ", self.evaluate(net, X_val, self.get_one_hot(y_val,2), params))
            print ("Test - ", self.evaluate(net, X_test, self.get_one_hot(y_test,2), params))




if __name__ == "__main__":

    #dirpath = '/home/andrea/Bureau/aitech/projects/bci/code/bci/experiment/reconstructed_2colours/'
    dirpath = '/home/andrea/Bureau/aitech/projects/bci/code/bci/experiment/2colours/'
    files = [f for f in os.listdir(dirpath) if f[-3:]=='csv' and len(f.split('_'))==4]
    print (files)
    # ##### Data format:
    # Datatype - float32 (both X and Y) <br>
    # X.shape - (#samples, 1 (kernel), #timepoints,  #channels) <br>
    # Y.shape - (#samples)

    ## Prepare data for model training


    prep= Data_prep()

    and_1, and_2, files1, files2 = prep.csv2arrays(file_dir=dirpath,cut_size=SIZE_NET,invert=False)
    #print('files col1: ',files1)
    #print ('size: ',SIZE_NET)
    and1_dstacked, and_y1= prep.list_2darrays_to_3d(and_1,0)
    and2_dstacked, and_y2 = prep.list_2darrays_to_3d(and_2,1)
    X = np.concatenate((and1_dstacked,and2_dstacked))
    y = np.concatenate((and_y1,and_y2))
    fileslist = files1 + files2

    X, y, fileslist = shuffle(X, y, fileslist, random_state=0)

    for i in range(len(y)):
        print (fileslist[i],'--',y[i])

    print( X.shape)
    ratio = 0.15
    len_test = int(ratio*len(y))
    len_train = len(y)-len_test
    X_train = X[:len_train,:,:,:]
    X_test = X[len_train:,:,:,:]
    y_train = y[:len_train]
    y_test = y[len_train:]
    filestest = fileslist[len_train:]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    def AddFileTOTestSet(X_test, y_test, files, testfile):

        print('file to test: ', testfile)
        expNo, name, col, *_ = testfile.split('_')
        kun_1 = np.genfromtxt('../' + testfile, delimiter=',').astype('float32')[:-1, :]
        kun_1 = kun_1[:, :SIZE_NET]
        Xt, yt = prep.list_2darrays_to_3d([kun_1], -1)
        print(X_test.shape)
        # (#samples, 1, #timepoints, #channels)
        array_dstack = np.array(Xt)
        array_dstack_reshaped = np.reshape(array_dstack, (1, 1, SIZE_NET, 16))

        X_test = np.concatenate((X_test, Xt))
        labelarr = np.array([int(col)-1])
        y_test = np.concatenate((y_test, labelarr))
        files.append(testfile)
        print('new shape: ',X_test.shape,'--',array_dstack_reshaped.shape)
        print('new shape label: ',len(y_test))
        return X_test,y_test,files


    testfile =  '1_testonline_1_20190202-163051.csv'
    X_test, y_test,filestest = AddFileTOTestSet(X_test, y_test, filestest, testfile)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1) #---> THAT'S WHY X_test acc > X_test acc!!!
    """

    ## initilise model and define optimizer and loss

    """


    net = EEGNet(SIZE_NET)

    print (net.forward(Variable(torch.Tensor(np.random.rand(5, 1, SIZE_NET, 16)))))

    criterion = nn.BCELoss()

    optimizer = optim.Adam(net.parameters())
    """

    ## train model using cross validation

    """
    start = time.time()
    eval = Evaluation()

    #eval.train_Kfold_validation(n_epochs=20)
    eval.train(n_epochs=50)
    print ('time: ',time.time()-start)
    eval.evaluate(net, X_test,eval.get_one_hot(y_test, 2), filestest, ['acc'], True, True)
    """

    ## save models state

    """
    save_path = './eeg_net_{}.pt'.format(date.today().strftime("%Y%m%d"))
    torch.save(net.state_dict(), save_path)


    '''
    TEST
    '''
    testfile = '1_testonline_1_20190202-163051.csv'#'1_testonline_2_20190202-162446.csv'
    print('file to test: ',testfile)
    kun_1 = np.genfromtxt( '../'+ testfile, delimiter=',').astype('float32')[:-1, :]
    kun_1 = kun_1[:, :SIZE_NET]
    X, y = prep.list_2darrays_to_3d([kun_1], -1)
    print(X.shape)
    # (#samples, 1, #timepoints, #channels)
    array_dstack = np.array(X)
    array_dstack_reshaped = np.reshape(array_dstack,(1, 1, SIZE_NET, 16))
    inputs = Variable(torch.from_numpy(array_dstack_reshaped))
    pred = net(inputs)
    print('prob: '+str(pred)) #Converted to probabilities
