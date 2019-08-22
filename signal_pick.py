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
import matplotlib.figure as figure

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

folder1 = "./cw_smooth_eval/"
folder2 = "./cw_l2_eval/"
folder3 = "./cw_l2smooth_0_01_eval/"
idx_file = 'data_select_i.csv'
# idx_file = './Subjective_AMT/distarr_A_N_0_360.csv'
# save_loc = './Subjective_AMT/Test_A_N_50/R' 
TRUTH = '0'
TARGET = '1'
data_select = genfromtxt(idx_file, delimiter=',')
g_label = classes[int(TRUTH)]
t_label = classes[int(TARGET)]

start = 0
end = 1
length = end-start
order_arr = np.zeros((length, 4), dtype=int)
count = 0

dist_arr = np.zeros((length, 5))

# for i in reversed(range(start,end)):
for i in range(start,end):
    idx = int(data_select[i, 3]) ## attention!
    idx = 3863
    order_arr[count, 0] = idx
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
    
    if (int(res_1) != int(TARGET)) or (int(res_2) != int(TARGET)) or (int(res_3) != int(TARGET)):
        print('Signal ' + str(idx) + ': some attack is not successful.') 
        continue
    
    ymax = np.max([sample, adv_sample_1, adv_sample_2, adv_sample_3])+1
    ymin = np.min([sample, adv_sample_1, adv_sample_2, adv_sample_3])-1
    
    # l2_dist = np.sum(np.square(adv_sample_2[:,0]-sample[:,0]))
    # print(l2_dist)
    # smooth_dist = np.var(np.diff(adv_sample_1[:,0]-sample[:,0], axis=0), axis=0)
    # l2_dist_part = np.sum(np.square(adv_sample_3[:,0]-sample[:,0]))
    # smooth_dist_part = np.var(np.diff(adv_sample_3[:,0]-sample[:,0], axis=0), axis=0)
    # dist_arr[count, 0] = idx
    # dist_arr[count, 1] = l2_dist
    # dist_arr[count, 2] = smooth_dist
    # dist_arr[count, 3] = l2_dist_part
    # dist_arr[count, 4] = smooth_dist_part
    
    # dist_arr = dist_arr[dist_arr[:,1].argsort()[::-1]]
    # count = count+1


    # plot with labels
    
    fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)
    
    axs[0].plot(sample[0:4000,:], color='black', label='Original signal')
#    axs[0].plot(adv_sample[:,:]-sample[:,:], color='green', label='perturbation')
    # axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
    axs[0].set_title('Original ECG signal, Class {}'.format(g_label))
    axs[0].set_ylim([ymin, ymax])
    axs[0].set_ylabel('Amplitude')
    
    axs[1].plot(adv_sample_1[0:4000,:], color='forestgreen', label='Adv signal_diff')
    # axs[1].set_title('Adv signal {}, Target {}'.format(r_label_1, t_label))
    axs[1].set_title('Adversarial ECG signal, Class {}, Metric: Smoothness'.format(r_label_1, t_label))
    axs[1].set_ylim([ymin, ymax])
    axs[1].set_ylabel('Amplitude')
    
    axs[2].plot(adv_sample_2[0:4000,:], color='forestgreen', label='Adv signal_l2')
    # axs[2].set_title('Adv signal {}, Target {}'.format(r_label_2, t_label))
    axs[2].set_title('Adversarial ECG signal, Class {}, Metric: L2-norm'.format(r_label_1, t_label))
    axs[2].set_ylim([ymin, ymax])
    axs[2].set_ylabel('Amplitude')
    
    axs[3].plot(adv_sample_3[0:4000,:], color='forestgreen', label='Adv signal_l2_diff')
    # axs[3].set_title('Adv signal {}, Target {}'.format(r_label_3, t_label))
    axs[3].set_title('Adversarial ECG signal, Class {}, Metric: Smoothness+L2-norm'.format(r_label_1, t_label))
    axs[3].set_ylim([ymin, ymax])
    axs[3].set_xlabel('Sample Index')
    axs[3].set_ylabel('Amplitude')
    fig.tight_layout()
     
    
    ###################### plot for subjective test(offline) #################################
    
    order = np.random.permutation(3) + 1
    order_arr[count, 1:4] = order
    fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)
    axs[0].plot(sample[0:4000,:], color='black', linewidth=2)
    axs[0].set_title('Reference signal')
    axs[0].set_ylim([ymin, ymax])
    axs[0].get_yaxis().set_visible(False)
    axs[0].get_xaxis().set_visible(False)
    
    axs[order[0]].plot(adv_sample_1[0:4000,:], color='forestgreen', linewidth=2)
    axs[1].set_title('Option A')
    axs[1].set_ylim([ymin, ymax])
    axs[1].get_yaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    
    axs[order[1]].plot(adv_sample_2[0:4000,:], color='forestgreen', linewidth=2)
    axs[2].set_title('Option B')
    axs[2].set_ylim([ymin, ymax])
    axs[2].get_yaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)
    
    axs[order[2]].plot(adv_sample_3[0:4000,:], color='forestgreen', linewidth=2)
    axs[3].set_title('Option C')
    axs[3].set_ylim([ymin, ymax])
    axs[3].set_xlabel('index')
    axs[3].get_yaxis().set_visible(False)
    axs[3].get_xaxis().set_visible(False)
    #axs[3].set_axis_off()
    fig.tight_layout()
    
    fig.savefig('./Subjective_offline/R' + str(idx) + '_all.png',bbox_inches='tight')
    
    ##################### plot for subjective test(AMT) #####################################
    
    num = count
    w, h = figure.figaspect(0.2)
    fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    axs.plot(adv_sample_1[0:4000,:], color='forestgreen',linewidth=2)
    axs.set_ylim([ymin, ymax])
    axs.get_yaxis().set_visible(False)
    axs.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_loc + str(num) + '_' + str(idx) + '_' + g_label + '_' + t_label + '_diff.png',bbox_inches='tight')
    
    
    w, h = figure.figaspect(0.2)
    fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    axs.plot(adv_sample_2[0:4000,:], color='forestgreen',linewidth=2)
    axs.set_ylim([ymin, ymax])
    axs.set_ylabel('signal value')
    axs.get_yaxis().set_visible(False)
    axs.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_loc + str(num) + '_' + str(idx) + '_' + g_label + '_' + t_label + '_l2.png',bbox_inches='tight')
    
    
    w, h = figure.figaspect(0.2)
    fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    axs.plot(adv_sample_3[0:4000,:], color='forestgreen',linewidth=2)
    axs.set_ylim([ymin, ymax])
    axs.set_ylabel('signal value')
    axs.get_yaxis().set_visible(False)
    axs.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_loc + str(num) + '_' + str(idx) + '_' + g_label + '_' + t_label + '_l2diff.png',bbox_inches='tight')
    
    
    w, h = figure.figaspect(0.2)
    fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    axs.plot(sample[0:4000,:], color='black', linewidth=2)
    axs.set_ylim([ymin, ymax])
    axs.set_ylabel('signal value')
    axs.get_yaxis().set_visible(False)
    axs.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_loc + str(num) + '_' + str(idx) + '_' + g_label + '_' + t_label + '_original.png',bbox_inches='tight')
    
    count +=1


# np.savetxt('./Subjective_AMT/distarr_'+ g_label + '_' + t_label + '_0_360.csv', dist_arr, delimiter=',')   

# np.savetxt('./Subjective_offline/order_'+ g_label + '_' + t_label + '_' + str(start) + '_' + str(end-1) + '.csv', order_arr, delimiter=',')

plt.close('all')
