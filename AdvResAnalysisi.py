#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:34:36 2018

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
    y = filtfilt(b, a, data, method="gust")
    return y


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

#--- parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']
#---filter para
lowcut = 0.01
highcut = 50
order_val = 3

data_select = genfromtxt('data_select.csv', delimiter=',')

#--- loading model and prepare wrapper
keras.layers.core.K.set_learning_phase(0)
sess = tf.Session()
K.set_session(sess)
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

fid = 6755
record = "A{:05d}".format(fid)
local_filename = "./training_raw/"+record
print('Loading record {}'.format(record))    
mat_data = scipy.io.loadmat(local_filename)
#data = mat_data['val'].squeeze()
data = mat_data['val']
print(data.shape)
#--- Processing data
data = preprocess(data, WINDOW_SIZE)
sample=np.float32(data)
sample = np.reshape(sample, (9000,1))

adv_sample = genfromtxt('./cwsdtw_eval/R' + str(fid) + '_0_1_1.csv', delimiter=',')
adv_sample = np.float32(adv_sample)
adv_sample = np.reshape(adv_sample, (9000,1))

#--- FFT
#yf = scipy.fftpack.fft(adv_sample)
#xf = np.linspace(0.0, FS/2.0, 9000//2)
#plt.plot(xf, 2.0/9000 * np.abs(yf[:9000//2]))
#--- Apply filter
adv_sample_filt = butter_bandpass_filter(adv_sample, lowcut, highcut, FS, order=order_val)
sample_filt = butter_bandpass_filter(sample, lowcut, highcut, FS, order=order_val)

#--- Check prediction
prob = model.predict(np.reshape(sample, (1, 9000, 1)))
ann = np.argmax(prob)
print(ann)

#prob = model.predict(np.reshape(sample_filt, (1, 9000, 1)))
#ann = np.argmax(prob)
#print(ann)

prob = model.predict(np.reshape(adv_sample, (1, 9000, 1)))
ann = np.argmax(prob)
print(ann)

#prob = model.predict(np.reshape(adv_sample_filt, (1, 9000, 1)))
#ann = np.argmax(prob)
#print(ann)

fig, axs = plt.subplots(2, 1, figsize=(50,40), sharex=True)
axs[0].plot(sample[0:3000,:], label='original sample')
#axs[0].plot(sample_filt, color='salmon', label='filtered original sample')
axs[0].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1), ncol=2)
axs[0].set_ylabel('signal value')
axs[1].plot(adv_sample[0:3000,:], label='adv sample')
#axs[1].plot(adv_sample_filt, color='salmon', label='filtered adv sample')
axs[1].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1), ncol=2)
axs[1].set_xlabel('sample index')
axs[1].set_ylabel('signal value')