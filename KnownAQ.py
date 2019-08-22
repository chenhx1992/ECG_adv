#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:15:24 2018

@author: chenhx1992
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.figure as figure
from numpy import genfromtxt

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

data_select_N = genfromtxt('data_select_N.csv', delimiter=',')
data_select_A = genfromtxt('data_select_A.csv', delimiter=',')
data_select_O = genfromtxt('data_select_O.csv', delimiter=',')
data_select_i = genfromtxt('data_select_i.csv', delimiter=',')

idx = 19
record = "A{:05d}".format(idx)
local_filename = "./training_raw/"+record
print('Loading record {}'.format(record))    
mat_data = scipy.io.loadmat(local_filename)
#data = mat_data['val'].squeeze()
data = mat_data['val']
#--- Processing data
data = preprocess(data, WINDOW_SIZE)
sample = np.reshape(data, (9000,1))

noise1 = np.random.rand(9000,1)
new1 = sample + noise1 

noise2 = np.random.rand(9000,1) * 0.5
new2 = sample + noise2

ymax = np.max(sample)+1
ymin = np.min(sample)-1

## plot all with random order
order = np.random.permutation(3) + 1
print(order)
fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)
#fig.suptitle('Test index: 1')
axs[0].plot(sample[0:4000,:], color='black', linewidth=2)
axs[0].set_title('Reference signal')
axs[0].set_ylim([ymin, ymax])
axs[0].set_ylabel('signal value')
axs[0].get_yaxis().set_visible(False)
#axs[0].set_axis_off()

axs[order[0]].plot(new1[0:4000,:], color='forestgreen', linewidth=2)
axs[order[0]].set_title('Option A')
axs[order[0]].set_ylim([ymin, ymax])
#    axs[1].set_xlabel('index')
axs[order[0]].set_ylabel('signal value')
axs[order[0]].get_yaxis().set_visible(False)
#axs[1].set_axis_off()

axs[order[1]].plot(new2[0:4000,:], color='forestgreen', linewidth=2)
axs[order[1]].set_title('Option B')
axs[order[1]].set_ylim([ymin, ymax])
axs[order[1]].set_ylabel('signal value')
axs[order[1]].get_yaxis().set_visible(False)
#axs[2].set_axis_off()

axs[order[2]].plot(sample[0:4000,:], color='forestgreen', linewidth=2)
axs[order[2]].set_title('Option C')
axs[order[2]].set_ylim([ymin, ymax])
axs[order[2]].set_xlabel('index')
axs[order[2]].set_ylabel('signal value')
axs[order[2]].get_yaxis().set_visible(False)
axs[order[2]].get_xaxis().set_visible(False)
#axs[3].set_axis_off()
fig.tight_layout()



## plot each one
idx = 5
fig.savefig('./KAQ/KAQ_' + str(idx) + '_all.png',bbox_inches='tight')

w, h = figure.figaspect(0.1)
fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
axs.plot(new2[0:4000,:], color='forestgreen', label='Original signal',linewidth=2)
#axs.plot(sample[0:4000,:], color='black', label='Original signal',linewidth=2)
#axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
axs.set_ylim([ymin, ymax])
#ratio=0.15
#xleft, xright = axs.get_xlim()
#ybottom, ytop = axs.get_ylim()
#axs.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
axs.get_yaxis().set_visible(False)
axs.set_axis_off()
fig.tight_layout()
fig.savefig('./KAQ/KAQ_' +str(idx) + '_F2.png',bbox_inches='tight')


w, h = figure.figaspect(0.1)
fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
axs.plot(new1[0:4000,:], color='forestgreen', label='Original signal',linewidth=2)
#axs.plot(sample[0:4000,:], color='black', label='Original signal',linewidth=2)
#axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
axs.set_ylim([ymin, ymax])
#ratio=0.15
#xleft, xright = axs.get_xlim()
#ybottom, ytop = axs.get_ylim()
axs.set_ylabel('signal value')
#axs.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
axs.get_yaxis().set_visible(False)
axs.set_axis_off()
fig.tight_layout()
fig.savefig('./KAQ/KAQ_' + str(idx) + '_F1.png',bbox_inches='tight')


w, h = figure.figaspect(0.1)
fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
axs.plot(sample[0:4000,:], color='forestgreen', label='Original signal',linewidth=2)
#axs.plot(sample[0:4000,:], color='black', label='Original signal',linewidth=2)
#axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
axs.set_ylim([ymin, ymax])
#ratio=0.15
#xleft, xright = axs.get_xlim()
#ybottom, ytop = axs.get_ylim()
axs.set_ylabel('signal value')
#axs.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
axs.get_yaxis().set_visible(False)
axs.set_axis_off()
fig.tight_layout()
fig.savefig('./KAQ/KAQ_' + str(idx) + '_T.png',bbox_inches='tight')


w, h = figure.figaspect(0.1)
fig, axs = plt.subplots(1, 1, figsize=(w, h), frameon=False)
axs.plot(sample[0:4000,:], color='black', label='Original signal',linewidth=2)
#axs[0].set_title('Original signal {}, Index {}'.format(g_label, str(idx)))
axs.set_ylim([ymin, ymax])
#ratio=0.15
#xleft, xright = axs.get_xlim()
#ybottom, ytop = axs.get_ylim()
axs.set_ylabel('signal value')
#axs.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
axs.get_yaxis().set_visible(False)
axs.set_axis_off()
fig.tight_layout()
fig.savefig('./KAQ/KAQ_' + str(idx) + '.png',bbox_inches='tight')
