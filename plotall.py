#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:41:08 2018

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

count = 0
folder = "./cw_smooth_eval/"
for file in os.listdir(folder):
    
    adv_sample = genfromtxt(folder + file, delimiter=',')
    adv_sample = np.reshape(adv_sample, (9000,1))
    para_list = file.rstrip('.csv')
    para_list = para_list.lstrip('R')
    idx, groundtruth, target, result = para_list.split('_')
#    print(idx, groundtruth, target, result)
    g_label = classes[int(groundtruth)]
    t_label = classes[int(target)]
    r_label = classes[int(result)]
    
    fid = int(idx)
    record = "A{:05d}".format(fid)
    local_filename = "./training_raw/"+record
    print('Loading record {}'.format(record))    
    mat_data = scipy.io.loadmat(local_filename)
    #data = mat_data['val'].squeeze()
    data = mat_data['val']
#    print(data.shape)
    #--- Processing data
    data = preprocess(data, WINDOW_SIZE)
    
    sample = np.reshape(data, (9000,1))
    
    ymax = np.max(adv_sample)+1
    ymin = np.min(adv_sample)-1
    
#    dist = np.var(np.diff(adv_sample[0,:]-sample[0,:], axis=0), axis=0)
    
    plt.ioff()
    fig, axs = plt.subplots(2, 1, figsize=(80,40), sharex=True)
    
    axs[0].plot(sample[:,:], color='black', label='Original signal')
#    axs[0].plot(adv_sample[:,:]-sample[:,:], color='green', label='perturbation')
    axs[0].set_title('Original signal {}, Index {}'.format(g_label, fid))
    axs[0].set_ylim([ymin, ymax])
    axs[0].set_ylabel('signal value')
#    axs[0].legend(loc='upper right', frameon=False)
    
    axs[1].plot(adv_sample[:,:], color='skyblue', label='Adv signal')
    axs[1].set_title('Adv signal {}, Target {}'.format(r_label, t_label))
    axs[1].set_ylim([ymin, ymax])
    axs[1].set_xlabel('index')
    axs[1].set_ylabel('signal value')
#    axs[1].legend(loc='upper right', frameon=False)
    
    plt.show()
    
    print('Save file', file.rstrip('.csv'))
    fig.savefig('./fig/'+ file.rstrip('.csv') + '.png', dpi=fig.dpi)
    
    count +=1
#    if count > 2:
#        break

print(count)
  
    
    
