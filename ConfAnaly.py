#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:58:40 2018

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
import matplotlib.figure as figure
import os
import glob

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

def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

#--- loading model and prepare wrapper
keras.layers.core.K.set_learning_phase(0)
sess = tf.Session()
K.set_session(sess)
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')


folder1 = "./cw_smooth_eval/"
folder2 = "./cw_l2_eval/"
folder3 = "./cw_l2smooth_0_01_eval/"
idx_file = 'data_select_i.csv'
# idx_file = './Subjective_AMT/distarr_A_N_0_360.csv'
# save_loc = './Subjective_AMT/Test_A_N_50/R' 
TRUTH = '3'
TARGET = '2'
data_select = genfromtxt(idx_file, delimiter=',')
g_label = classes[int(TRUTH)]
t_label = classes[int(TARGET)]

start = 0
end = 220
length = end-start

orginal_arr = np.zeros((length, 1), dtype=float)
advl2_arr = np.zeros((length, 1), dtype=float)
advdiff_arr = np.zeros((length, 1), dtype=float)
advl2diff_arr = np.zeros((length, 1), dtype=float)

# for i in reversed(range(start,end)):
for i in range(start,end):
    idx = int(data_select[i, 3]) ## attention!
    
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
    
    file3 = glob.glob(folder3 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_3 = genfromtxt(file3, delimiter=',')
    adv_sample_3 = np.reshape(adv_sample_3, (9000,1))
    res_3 = file3[-5]
    r_label_3 = classes[int(res_3)]
    
    # if (int(res_1) != int(TARGET)) or (int(res_2) != int(TARGET)) or (int(res_3) != int(TARGET)):
    #     print('Signal ' + str(idx) + ': some attack is not successful.') 
    #     continue
        
    prob_o = model.predict(np.reshape(sample, (1,9000,1)))
    ann_o = np.argmax(prob_o)
    if str(ann_o) == TRUTH:
        orginal_arr[i, 0] = np.max(prob_o)
    
    # adv_sample_1 = zero_mean(adv_sample_1)
    prob_diff = model.predict(np.reshape(adv_sample_1, (1, 9000,1)))
    ann_diff = np.argmax(prob_diff)
    if str(ann_diff) == TARGET:
        advdiff_arr[i, 0] = np.max(prob_diff)
    
    # adv_sample_2 = zero_mean(adv_sample_2)
    prob_l2 = model.predict(np.reshape(adv_sample_2, (1, 9000,1)))
    ann_l2 = np.argmax(prob_l2)
    if str(ann_l2) == TARGET:
        advl2_arr[i, 0] = np.max(prob_l2)
        
    
    # adv_sample_3 = zero_mean(adv_sample_3)
    prob_l2diff = model.predict(np.reshape(adv_sample_3,(1, 9000, 1)))
    ann_l2diff = np.argmax(prob_l2diff)
    if str(ann_l2diff) == TARGET:
        advl2diff_arr[i, 0] = np.max(prob_l2diff)
        
orginal_arr = -orginal_arr

np.savetxt('./paper_fig/predictprob_' + TRUTH + '_' + TARGET + '_original.csv', orginal_arr, delimiter=',')
np.savetxt('./paper_fig/predictprob_' + TRUTH + '_' + TARGET + '_l2.csv',advl2_arr, delimiter=',')
np.savetxt('./paper_fig/predictprob_' + TRUTH + '_' + TARGET + '_diff.csv',advdiff_arr, delimiter=',')
np.savetxt('./paper_fig/predictprob_' + TRUTH + '_' + TARGET + '_l2diff.csv',advl2diff_arr, delimiter=',')

plt_start = 0
plt_end = 100
x = np.arange(plt_end-plt_start)
fig, axs = plt.subplots(1, 1, figsize=(160,40))
axs.bar(x, orginal_arr[plt_start:plt_end,0], width=0.9, color='darkgrey')
axs.bar(x-0.3, advl2_arr[plt_start:plt_end,0], width=0.3, color='lightblue')
axs.bar(x, advdiff_arr[plt_start:plt_end,0], width=0.3, color='greenyellow')
axs.bar(x+0.3, advl2diff_arr[plt_start:plt_end,0], width=0.3, color='salmon')
axs.set_ylim([-1,+1])
axs.set_ylabel('Prediction Confidence')
axs.set_xlabel('Sample Index')
fig.tight_layout()
fig.savefig('./paper_fig/predictprob_' + TRUTH + '_' + TARGET + '.png',bbox_inches='tight')