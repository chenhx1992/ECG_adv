#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:15:13 2018

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
save_loc = './Subjective_prof_4/R' 

f = open(idx_file, 'r')

count = 1
for line in f:
    print(line)
    idx, source, target = line.rstrip('\n').split('_')
    TRUTH = classes.index(source)
    TARGET = classes.index(target)
    # print(TRUTH, TARGET)
    
    g_label=source
    t_label=target
    TRUTH = str(TRUTH)
    TARGET = str(TARGET)
    idx = int(idx.lstrip('R'))
    
    record = "A{:05d}".format(idx)
    local_filename = "./training_raw/"+record
    print('Loading record {}'.format(record))
    headerfile = wfdb.rdheader(local_filename) 
    # print(headerfile.__dict__)
    adc_gain = headerfile.__dict__['adc_gain'][0]
    initial_value = headerfile.__dict__['init_value'][0]
    sig_len = headerfile.__dict__['sig_len']
    
    if (sig_len != 9000):
        print('No length 9000')
    mat_data = scipy.io.loadmat(local_filename)
    #data = mat_data['val'].squeeze()
    data = mat_data['val']
    #--- Processing data
    pre_data, std = preprocess(data, WINDOW_SIZE)
    sample = np.reshape(pre_data, (9000,1))
    scale = adc_gain / std
    sample = sample / scale - initial_value/adc_gain
    
    file1 = glob.glob(folder1 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_1 = genfromtxt(file1, delimiter=',')
    adv_sample_1 = np.reshape(adv_sample_1, (9000,1))
    adv_sample_1 = adv_sample_1 /scale - initial_value/adc_gain
    res_1 = file1[-5]
    r_label_1 = classes[int(res_1)]
    
    
    file2 = glob.glob(folder2 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_2 = genfromtxt(file2, delimiter=',')
    adv_sample_2 = np.reshape(adv_sample_2, (9000,1))
    adv_sample_2 = adv_sample_2 / scale - initial_value/adc_gain
    res_2 = file2[-5]
    r_label_2 = classes[int(res_2)]
    
    file3 = glob.glob(folder3 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
    adv_sample_3 = genfromtxt(file3, delimiter=',')
    adv_sample_3 = np.reshape(adv_sample_3, (9000,1))
    adv_sample_3 = adv_sample_3 / scale - initial_value/adc_gain
    res_3 = file3[-5]
    r_label_3 = classes[int(res_3)]
    
    ymax = np.max([sample, adv_sample_1, adv_sample_2, adv_sample_3])
    ymin = np.min([sample, adv_sample_1, adv_sample_2, adv_sample_3])
    print(ymin, ymax)
    
    # plot with labels
    
#     fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)
#     length = 9000
#     axs[0].plot(sample[0:length,:], color='black', label='Original signal')
# #    axs[0].plot(adv_sample[:,:]-sample[:,:], color='green', label='perturbation')
#     axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
#     axs[0].set_ylim([ymin, ymax])
#     axs[0].set_ylabel('signal value')
    
#     axs[1].plot(adv_sample_1[0:length,:], color='forestgreen', label='Adv signal_diff')
#     axs[1].set_title('Adv signal {}, Target {}'.format(r_label_1, t_label))
#     axs[1].set_ylim([ymin, ymax])
#     axs[1].set_ylabel('signal value')
    
#     axs[2].plot(adv_sample_2[0:length,:], color='forestgreen', label='Adv signal_l2')
#     axs[2].set_title('Adv signal {}, Target {}'.format(r_label_2, t_label))
#     axs[2].set_ylim([ymin, ymax])
#     axs[2].set_ylabel('signal value')
    
#     axs[3].plot(adv_sample_3[0:length,:], color='forestgreen', label='Adv signal_l2_diff')
#     axs[3].set_title('Adv signal {}, Target {}'.format(r_label_3, t_label))
#     axs[3].set_ylim([ymin, ymax])
#     axs[3].set_xlabel('index')
#     axs[3].set_ylabel('signal value')
#     fig.tight_layout()
    
    ##################### plot for subjective test(Doctor) #####################################
    length = 9000
    lw = 1
    # major_x_ticks = np.arange(0, 9300, 60)
    # minor_x_ticks = np.arange(0, 9300, 12)
    # major_x_ticklabels = np.arange(0, 31, 1)
    major_x_ticks = np.arange(0, 3100, 60)
    minor_x_ticks = np.arange(0, 3100, 12)
    major_x_ticklabels = np.arange(0, 11, 1)


    # margin = round((ymax-ymin)*0.1, 1)
    # lb = round(ymin) - margin
    # hb = round(ymax) + margin
    lb = math.floor(ymin-0.0001)
    hb = math.ceil(ymax+0.0001)
    lb = -1
    hb = 2
    
    major_y_ticks = np.arange(lb, hb+0.1, 0.5)
    minor_y_ticks = np.arange(lb, hb+0.1, 0.1)
    major_y_ticklabels = np.arange(lb, hb+0.1, 1)
    major_y_ticklabels =  np.round(major_y_ticklabels,2)
    # w, h = figure.figaspect(0.5)
    w = 750
    h = 10
    
    fig, axs = plt.subplots(3,1,figsize=(w, h), frameon=False)
    fig.subplots_adjust(hspace = .001)
    axs[0].plot(adv_sample_1[0:3000], color='black',linewidth=lw)
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
    
    axs[1].plot(adv_sample_1[3000:6000], color='black',linewidth=lw)
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
    
    axs[2].plot(adv_sample_1[6000:9000], color='black',linewidth=lw)
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
    fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_diff.png',dpi=330, bbox_inches='tight')
    
    fig, axs = plt.subplots(3, 1, figsize=(w, h), frameon=False)
    fig.subplots_adjust(hspace = .001)
    axs[0].plot(adv_sample_2[0:3000], color='black',linewidth=lw)
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
    
    axs[1].plot(adv_sample_2[3000:6000], color='black',linewidth=lw)
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
    
    axs[2].plot(adv_sample_2[6000:9000], color='black',linewidth=lw)
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
    fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2.png',dpi=330, bbox_inches='tight')
    
    fig, axs = plt.subplots(3, 1, figsize=(w, h), frameon=False)
    fig.subplots_adjust(hspace = .001)
    axs[0].plot(adv_sample_3[0:3000], color='black',linewidth=lw)
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
    
    axs[1].plot(adv_sample_3[3000:6000], color='black',linewidth=lw)
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
    
    axs[2].plot(adv_sample_3[6000:9000], color='black',linewidth=lw)
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
    fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2diff.png',dpi=330, bbox_inches='tight')
    
    
    fig, axs = plt.subplots(3, 1, figsize=(w, h), frameon=False)
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
    fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_original.png',dpi=330, bbox_inches='tight')
    
    ##################### plot for subjective test(doctor) one array #####################################
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_1[0:length,:], color='black',linewidth=lw)
    # axs.set_xticks(major_x_ticks)
    # axs.set_xticks(minor_x_ticks, minor=True)
    # axs.set_xticklabels([])
    # axs.set_yticks(major_y_ticks)
    # axs.set_yticks(minor_y_ticks, minor=True)
    # axs.set_yticklabels([])
    # # axs.set_xlabel('Time (second)')
    # # axs.set_ylabel('Amplitude (mV)')
    # axs.set_ylim([lb, hb])
    # axs.set_xlim([-30, 9030])
    # axs.grid(which='minor', color='salmon', axis='both', alpha=0.4)
    # axs.grid(which='major', color='salmon', axis='both', alpha=0.8)
    # axs.text(120, hb+2.5, '10 mm/mV', fontsize=10)
    # axs.text(120, hb+1.5, '25 mm/sec', fontsize=10)
    # axs.text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
    # axs.set_aspect(120)
    # fig.tight_layout()
    # plt.get_current_fig_manager().window.showMaximized()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_diff.png',dpi=330, bbox_inches='tight')
    
    
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_2[0:length,:], color='black',linewidth=lw)
    # axs.set_xticks(major_x_ticks)
    # axs.set_xticks(minor_x_ticks, minor=True)
    # axs.set_xticklabels([])
    # axs.set_yticks(major_y_ticks)
    # axs.set_yticks(minor_y_ticks, minor=True)
    # axs.set_yticklabels([])
    # # axs.set_xlabel('Time (second)')
    # # axs.set_ylabel('Amplitude (mV)')
    # axs.set_ylim([lb, hb])
    # axs.set_xlim([-30, 9030])
    # axs.grid(which='minor', color='salmon', axis='both', alpha=0.4)
    # axs.grid(which='major', color='salmon', axis='both', alpha=0.8)
    # axs.text(120, hb+2.5, '10 mm/mV', fontsize=10)
    # axs.text(120, hb+1.5, '25 mm/sec', fontsize=10)
    # axs.text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
    # axs.set_aspect(120)
    # fig.tight_layout()
    # plt.get_current_fig_manager().window.showMaximized()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2.png',dpi=330, bbox_inches='tight')
    
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_3[0:length,:], color='black',linewidth=lw)
    # axs.set_xticks(major_x_ticks)
    # axs.set_xticks(minor_x_ticks, minor=True)
    # axs.set_xticklabels([])
    # axs.set_yticks(major_y_ticks)
    # axs.set_yticks(minor_y_ticks, minor=True)
    # axs.set_yticklabels([])
    # # axs.set_xlabel('Time (second)')
    # # axs.set_ylabel('Amplitude (mV)')
    # axs.set_ylim([lb, hb])
    # axs.set_xlim([-30, 9030])
    # axs.grid(which='minor', color='salmon', axis='both', alpha=0.4)
    # axs.grid(which='major', color='salmon', axis='both', alpha=0.8)
    # axs.text(120, hb+2.5, '10 mm/mV', fontsize=10)
    # axs.text(120, hb+1.5, '25 mm/sec', fontsize=10)
    # axs.text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
    # axs.set_aspect(120)
    # fig.tight_layout()
    # plt.get_current_fig_manager().window.showMaximized()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2diff.png',dpi=330, bbox_inches='tight')
    
    
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(sample[0:length,:], color='black',linewidth=lw)
    # axs.set_xticks(major_x_ticks)
    # axs.set_xticks(minor_x_ticks, minor=True)
    # axs.set_xticklabels([])
    # axs.set_yticks(major_y_ticks)
    # axs.set_yticks(minor_y_ticks, minor=True)
    # axs.set_yticklabels([])
    # # axs.set_xlabel('Time (second)')
    # # axs.set_ylabel('Amplitude (mV)')
    # axs.set_ylim([lb, hb])
    # axs.set_xlim([-30, 9030])
    # axs.grid(which='minor', color='salmon', axis='both', alpha=0.4)
    # axs.grid(which='major', color='salmon', axis='both', alpha=0.8)
    # axs.text(120, hb+2.5, '10 mm/mV', fontsize=10)
    # axs.text(120, hb+1.5, '25 mm/sec', fontsize=10)
    # axs.text(120, hb+0.5, 'Sampling freq: 300Hz', fontsize=10)
    # axs.set_aspect(120)
    # fig.tight_layout()
    # plt.get_current_fig_manager().window.showMaximized()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_original.png',dpi=330, bbox_inches='tight')
    
    
    ##################### plot for subjective test(AMT) #####################################
    # length = 9000
    # lw = 1
    # w, h = figure.figaspect(0.2)
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_1[0:length,:], color='black',linewidth=lw)
    # axs.set_ylim([ymin, ymax])
    # axs.set_ylabel('signal value')
    # axs.get_yaxis().set_visible(False)
    # axs.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_diff.png',dpi=330, bbox_inches='tight')
    
    
    # w, h = figure.figaspect(0.2)
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_2[0:length,:], color='black',linewidth=lw)
    # axs.set_ylim([ymin, ymax])
    # axs.set_ylabel('signal value')
    # axs.get_yaxis().set_visible(False)
    # axs.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2.png',dpi=330, bbox_inches='tight')
    
    
    # w, h = figure.figaspect(0.2)
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(adv_sample_3[0:length,:], color='black',linewidth=lw)
    # axs.set_ylim([ymin, ymax])
    # axs.set_ylabel('signal value')
    # axs.get_yaxis().set_visible(False)
    # axs.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_l2diff.png',dpi=330, bbox_inches='tight')
    
    
    # w, h = figure.figaspect(0.2)
    # fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
    # axs.plot(sample[0:length,:], color='black', linewidth=lw)
    # axs.set_ylim([ymin, ymax])
    # axs.set_ylabel('signal value')
    # axs.get_yaxis().set_visible(False)
    # axs.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(save_loc + str(idx) + '_' + g_label + '_' + t_label + '_original.png',dpi=330,bbox_inches='tight')
    # plt.close('all')
    plt.close('all')
f.close()


import pandas as pd
import shutil
df = pd.DataFrame()
order = np.random.permutation(48) + 1
count=0
for file in sorted(os.listdir("./Subjective_prof_4/")):
    if file.endswith('.png'):
        print(order[count], file)
        df.loc[file, 'order'] = order[count]
        shutil.copy("./Subjective_prof_4/"+file, "./Subjective_prof_shuffle_4/pic_"+str(order[count])+'.png')
        count +=1
df.to_csv("./Subjective_prof_4/shuffle_order.csv")
