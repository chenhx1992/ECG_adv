#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:15:44 2018

@author: chenhx1992
"""

from keras.utils import plot_model
import keras.backend as K
import keras
from keras import backend
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras import metrics
import tensorflow as tf
import pydot
import h5py
from numpy import genfromtxt
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
import scipy.fftpack
import matplotlib.pyplot as plt

FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

#### Funtion definition
def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    
    tmp = np.zeros((1, maxlen))
#    print(x.shape)
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
#    print(x.shape)
    x = np.expand_dims(x, axis=2)  # required by Keras
#    print(x.shape)
    del tmp
    
    return x


import os
import glob

count = 0
folder1 = "./cw_smooth_eval/"
folder2 = "./cw_l2_eval/"
idx_file = 'data_select_A.csv'
TRUTH = '0'
TARGET = '1'
data_select = genfromtxt(idx_file, delimiter=',')
g_label = classes[int(TRUTH)]
t_label = classes[int(TARGET)]


for i in reversed(range(60,70)):
    idx = int(data_select[i, 3]) 
    
    record = "A{:05d}".format(idx)
    local_filename = "./training_raw/"+record
    print('Loading record {}'.format(record))    
    mat_data = scipy.io.loadmat(local_filename)
    #data = mat_data['val'].squeeze()
    data = mat_data['val']
    #--- Processing data
    data = preprocess(data, WINDOW_SIZE)
    sample = np.reshape(data, (9000,1))
    
    file1 = glob.glob(folder1 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_1 = genfromtxt(file1, delimiter=',')
    adv_sample_1 = np.reshape(adv_sample_1, (9000,1))
    res_1 = file1[-5]
    r_label_1 = classes[int(res_1)]
    
    
    file2 = glob.glob(folder2 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_2 = genfromtxt(file2, delimiter=',')
    adv_sample_2 = np.reshape(adv_sample_2, (9000,1))
    res_2 = file2[-5]
    r_label_2 = classes[int(res_2)]
    
    ymax = np.max(adv_sample_1)+1
    ymin = np.min(adv_sample_1)-1
    
#    dist = np.var(np.diff(adv_sample[0,:]-sample[0,:], axis=0), axis=0)
    
#    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(80,40), sharex=True)
    
    axs[0].plot(sample[0:4000,:], color='black', label='Original signal')
#    axs[0].plot(adv_sample[:,:]-sample[:,:], color='green', label='perturbation')
    axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
    axs[0].set_ylim([ymin, ymax])
    axs[0].set_ylabel('signal value')
#    axs[0].legend(loc='upper right', frameon=False)
    
    axs[1].plot(adv_sample_1[0:4000,:], color='forestgreen', label='Adv signal_diff')
    axs[1].set_title('Adv signal {}, Target {}'.format(r_label_1, t_label))
    axs[1].set_ylim([ymin, ymax])
    axs[1].set_xlabel('index')
    axs[1].set_ylabel('signal value')
#    axs[1].legend(loc='upper right', frameon=False)
    
    axs[2].plot(adv_sample_2[0:4000,:], color='forestgreen', label='Adv signal_l2')
    axs[2].set_title('Adv signal {}, Target {}'.format(r_label_2, t_label))
    axs[2].set_ylim([ymin, ymax])
    axs[2].set_xlabel('index')
    axs[2].set_ylabel('signal value')
#    axs[1].legend(loc='upper right', frameon=False)
    plt.show(block=True)
    
#    print('Save file', file.rstrip('.csv'))
#    fig.savefig('./fig/'+ file.rstrip('.csv') + '.png', dpi=fig.dpi)
    
    count +=1
#    if count > 2:
#        break

#print(count)