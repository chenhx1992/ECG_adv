#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:12:16 2018

@author: chenhx1992
"""
import scipy.io
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN

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

## Loading time serie signals
id = 6755
count = id-1
record = "A{:05d}".format(id)
local_filename = "./training_raw/"+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']
#print(data.shape)
data = preprocess(data, WINDOW_SIZE)

X_test=np.float32(data)

adv_sample_l2 = genfromtxt('./result_6755/R6755_0_1_1_l2.csv', delimiter=',')
adv_sample_1 = genfromtxt('./result_6755/R6755_0_1_1_l2_smooth_1.csv', delimiter=',')
adv_sample_0_1 = genfromtxt('./result_6755/R6755_0_1_1_l2_smooth_0.1.csv', delimiter=',')
adv_sample_0_01 = genfromtxt('./result_6755/R6755_0_1_1_l2_smooth_0.01.csv', delimiter=',')
adv_sample_0_001 = genfromtxt('./result_6755/R6755_0_1_1_l2_smooth_0.001.csv', delimiter=',')
adv_sample = genfromtxt('./result_6755/R6755_0_1_1_smooth.csv', delimiter=',')

ymin = -2.5
ymax = 6.5

fig, axs = plt.subplots(2, 1, figsize=(80,40), sharex=True, sharey=True)
axs[0].plot(X_test[0,0:4000,:], color='black', label='original')
axs[0].set_title('Original signal A')
axs[0].set_ylim([ymin, ymax])
axs[0].set_ylabel('signal value')
axs[0].legend()

axs[1].plot(X_test[0,0:4000,:], color='black', label='original')
axs[1].plot(adv_sample_l2[0:4000], color='salmon', label='adv l2')
axs[1].plot(adv_sample_1[0:4000], color='blue', label='adv l2+smooth, para=1')
axs[1].plot(adv_sample_0_1[0:4000], color='greenyellow', label='adv l2+smooth, para=0.1')
axs[1].plot(adv_sample_0_01[0:4000], color='gold', label='adv l2+smooth, para=0.01')
axs[1].plot(adv_sample_0_001[0:4000], color='plum', label='adv l2+smooth, para=0.001')
axs[1].plot(adv_sample[0:4000], color='lightblue', label='adv smooth')
axs[1].set_title('Adv signal N')
axs[1].set_ylim([ymin, ymax])
axs[1].set_xlabel('index')
axs[1].set_ylabel('signal value')
axs[1].legend(loc='upper right')

#fig.savefig('p9.png',dpi=fig.dpi)