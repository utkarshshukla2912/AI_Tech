from scipy.signal import coherence
import matplotlib.pyplot as plt
from scipy.stats import moment
from tqdm import tqdm
from entropy import *
import pandas as pd
import numpy as np
import warnings
import logging
import pyeeg
import nolds
import math
import pywt
import glob
import json
import os
warnings.filterwarnings('ignore')


class featureExtractionPipeline:

    def __init__(self,channel_data,fs = 128):
        self.fs = fs
        self.bands = [0.1,4,8,12,30,50]
        self.channel_data = channel_data

    def BandPower(self):
        resp = pyeeg.bin_power(self.channel_data,self.bands,self.fs)
        return np.array(resp).flatten()

    def SpectralEntropy(self):
        b = pyeeg.bin_power(self.channel_data,self.bands,self.fs)
        resp = pyeeg.spectral_entropy(self.channel_data,self.bands,self.fs,Power_Ratio=b)
        resp = [0 if math.isnan(x) else x for x in resp]
        return np.array(resp)

    def CorrelationDimension(self):
        resp = nolds.corr_dim(self.channel_data,1)
        return np.array([resp])

    def DFA(self):
    	resp = pyeeg.dfa(self.channel_data)
    	return np.array([resp])

    def FirstDiff(self):
    	resp = pyeeg.first_order_diff(self.channel_data)
    	return resp

    def Hjorth(self):
    	resp = pyeeg.hjorth(self.channel_data)
    	return np.array(resp)

    def Hurst(self):
    	resp = pyeeg.hurst(self.channel_data)
    	return np.array([resp])

    def Mean(self):
    	resp = np.mean(self.channel_data)
    	return np.array([resp])

    def PFD(self):
    	resp = pyeeg.pfd(self.channel_data)
    	return np.array([resp])

    def Power(self):
    	F = np.fft.fft(self.channel_data)
    	P = F * np.conjugate(F)
    	resp = sum(P)
    	return np.array([abs(resp)])

    def Std(self):
    	resp = np.std(self.channel_data)
    	return np.array([resp])
    
    def skewness(self):
        resp = moment(self.channel_data, moment=3)
        return np.array([resp])
    
    def kurtosis(self):
        resp = moment(self.channel_data, moment=4)
        return np.array([resp])

    def dwt_features(self,coef_arr):
        s = np.sum(coef_arr)
        mean = np.mean(coef_arr)
        mi = min(coef_arr)
        ma = max(coef_arr)
        std = np.std(coef_arr)
        skewness = moment(coef_arr, moment=3)
        kurtosis = moment(coef_arr, moment=4)
        energy = sum(map(lambda x:x*x,coef_arr))
        #return [s,mean,skewness,energy]
        return [s,mean,mi,ma,std,skewness,kurtosis,energy]

    def DWT(self):
        resp = pywt.dwt(self.channel_data, 'db1')
        feature_ = self.dwt_features(resp[1])
        return np.array(feature_)
    
    def sampleEntropy(self):
        resp = nolds.sampen(self.channel_data)
        return np.array([resp])
    
    #def approximateEntropy(self)

    def runPipeline(self):
        '''
        Features to consider:
        Mean, STD, skewness, kurtosis, sample entropy,
        approximate entropy, wavelet tranform, Hjorth parameter,
        PSD, cumulative mean square density, waveform length, 
        energy spectral density
        
        Features yet to figure out:
        
        2. approximate entropy
        3. wavelet tranfrom
        4. Power spectral density
        5. cumulative mean square density
        6. waveform length
        7. energy spectral density
        
        '''
        
        
        features = np.array([
        self.Mean(),
        self.Std(),
        self.skewness(),
        self.kurtosis(),
        self.sampleEntropy(),
        self.DWT(),
        self.Hjorth(),
        self.Power(),
        self.SpectralEntropy(),
        ])
        feature_list = []
        for i in features:
            feature_list += i.tolist()
        return(feature_list)
        return(0)






