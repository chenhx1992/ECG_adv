#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:17:00 2018

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
import wfdb
import math

FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

#### Funtion definition
def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    std = np.std(x)
    x = x / std
    
    tmp = np.zeros((1, maxlen))
#    print(x.shape)
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
#    print(x.shape)
    x = np.expand_dims(x, axis=2)  # required by Keras
#    print(x.shape)
    del tmp
    
    return x, std

folder1 = "./cw_smooth_eval/"
folder2 = "./cw_l2_eval/"
folder3 = "./cw_l2smooth_0_01_eval/"
idx_file = './Subjective_prof_3/name.txt'

# f = open(idx_file, 'r')

count = 1

# idx, source, target = line.rstrip('\n').split('_')
idx = 511
source = 'O'
target = 'N'
TRUTH = classes.index(source)
TARGET = classes.index(target)
print(TRUTH, TARGET)

g_label=source
t_label=target
TRUTH = str(TRUTH)
TARGET = str(TARGET)
# idx = int(idx.lstrip('R'))

record = "A{:05d}".format(idx)
local_filename = "./training_raw/"+record
print('Loading record {}'.format(record))
headerfile = wfdb.rdheader(local_filename) 
# print(headerfile.__dict__)
adc_gain = headerfile.__dict__['adc_gain'][0]
initial_value = headerfile.__dict__['init_value'][0]
sig_len = headerfile.__dict__['sig_len']

if (sig_len!=9000):
    print('No length 9000')
print(adc_gain)
mat_data = scipy.io.loadmat(local_filename)
#data = mat_data['val'].squeeze()
data = mat_data['val']
#--- Processing data
pre_data, std = preprocess(data, WINDOW_SIZE)
sample = np.reshape(pre_data, (9000,))
scale = adc_gain / std 

data = np.reshape(data, (9000,))
pre_data = np.reshape(pre_data, (9000,))
# plt.plot((data - initial_value)/adc_gain, 'blue')
# plt.plot(pre_data/scale - initial_value/adc_gain, 'red')

file1 = glob.glob(folder3 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
adv_sample_1 = genfromtxt(file1, delimiter=',')
adv_sample_1 = np.reshape(adv_sample_1, (9000,))
res_1 = file1[-5]
r_label_1 = classes[int(res_1)]

# data = np.reshape(data, (9000,))
# new_data = data - initial_value
# new_data = new_data / adc_gain
# sample = new_data
# plt.plot(new_data)

sample = adv_sample_1 /scale - initial_value/adc_gain
sample = -sample
length = 9000
lw = 1
major_x_ticks = np.arange(0, 3100, 60)
minor_x_ticks = np.arange(0, 3100, 12)
major_x_ticklabels = np.arange(0, 11, 1)

# margin = round((ymax-ymin)*0.1, 1)
# lb = round(ymin) - margin
# hb = round(ymax) + margin
hb = 2
lb = -1
major_y_ticks = np.arange(lb, hb+0.1, 0.5)
minor_y_ticks = np.arange(lb, hb+0.1, 0.1)
major_y_ticklabels = np.arange(lb, hb+0.1, 1)
major_y_ticklabels =  np.round(major_y_ticklabels,2)
# w, h = figure.figaspect(0.5)

# plt.ioff()
# plt.ion()
# fig, axs = plt.subplots(1, 1, figsize=(750, 10), frameon=False)
fig, axs = plt.subplots(3,1,figsize=(750, 10), frameon=False)
fig.subplots_adjust(hspace = .001)
axs[0].plot(sample[0:3000], color='black',linewidth=lw)
axs[0].set_xticks(major_x_ticks)
axs[0].set_xticks(minor_x_ticks, minor=True)
axs[0].set_xticklabels([])
axs[0].set_yticks(major_y_ticks)
axs[0].set_yticks(minor_y_ticks, minor=True)
axs[0].set_yticklabels([])
# axs.get_yaxis().set_visible(False)
# axs.get_xaxis().set_visible(False)
# axs.set_xlabel('Time (second)')
# axs.set_ylabel('Amplitude (mV)')
axs[0].set_ylim([lb, hb])
axs[0].set_xlim([0, 3000])
axs[0].grid(which='minor', color='salmon', axis='both', alpha=0.4)
axs[0].grid(which='major', color='salmon', axis='both', alpha=0.8)
axs[0].text(100, hb+0.7, '10 mm/mV', fontsize=14)
axs[0].text(400, hb+0.7, '25 mm/sec', fontsize=14)
axs[0].text(700, hb+0.7, 'Sampling freq: 300Hz', fontsize=14)
axs[0].text(100, hb+0.1, '00:00', fontsize=14)
axs[0].set_aspect(120)

axs[1].plot(sample[3000:6000], color='black',linewidth=lw)
axs[1].set_xticks(major_x_ticks)
axs[1].set_xticks(minor_x_ticks, minor=True)
axs[1].set_xticklabels([])
axs[1].set_yticks(major_y_ticks)
axs[1].set_yticks(minor_y_ticks, minor=True)
axs[1].set_yticklabels([])
# axs.get_yaxis().set_visible(False)
# axs.get_xaxis().set_visible(False)
# axs.set_xlabel('Time (second)')
# axs.set_ylabel('Amplitude (mV)')
axs[1].set_ylim([lb, hb])
axs[1].set_xlim([0, 3000])
axs[1].grid(which='minor', color='salmon', axis='both', alpha=0.4)
axs[1].grid(which='major', color='salmon', axis='both', alpha=0.8)
axs[1].text(100, hb+0.1, '00:10', fontsize=14)
# axs[1].text(120, hb+2.5, '10 mm/mV', fontsize=10)
# axs[1].text(120, hb+1.5, '25 mm/sec', fontsize=10)
# axs[1].text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
axs[1].set_aspect(120)

axs[2].plot(sample[6000:9000], color='black',linewidth=lw)
axs[2].set_xticks(major_x_ticks)
axs[2].set_xticks(minor_x_ticks, minor=True)
axs[2].set_xticklabels([])
axs[2].set_yticks(major_y_ticks)
axs[2].set_yticks(minor_y_ticks, minor=True)
axs[2].set_yticklabels([])
# axs.get_yaxis().set_visible(False)
# axs.get_xaxis().set_visible(False)
# axs.set_xlabel('Time (second)')
# axs.set_ylabel('Amplitude (mV)')
axs[2].set_ylim([lb, hb])
axs[2].set_xlim([0, 3000])
axs[2].grid(which='minor', color='salmon', axis='both', alpha=0.4)
axs[2].grid(which='major', color='salmon', axis='both', alpha=0.8)
# axs[2].text(120, hb+2.5, '10 mm/mV', fontsize=10)
# axs[2].text(120, hb+1.5, '25 mm/sec', fontsize=10)
# axs[2].text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
axs[2].set_aspect(120)
axs[2].text(100, hb+0.1, '00:20', fontsize=14)
plt.subplots_adjust(wspace=None, hspace=None)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,wspace=0.35)
fig.tight_layout()
plt.get_current_fig_manager().window.showMaximized()
fig.savefig('./Subjective_prof_4/R' + str(idx) + '_' + g_label + '_' + t_label + '_l2diff.png',dpi=330, bbox_inches='tight')
    