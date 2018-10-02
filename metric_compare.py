#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:39:47 2018

@author: chenhx1992
"""

import matplotlib.pyplot as plt
from numpy import genfromtxt
import scipy.io
import numpy as np

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

dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

id = 6755
count = id-1
record = "A{:05d}".format(id)
local_filename = "./training_raw/"+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']
print(data.shape)
data = preprocess(data, WINDOW_SIZE)
X_test=np.float32(data)

adv_sample_g1_0 = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_g1_0.csv', delimiter=',')
adv_sample_g1_0 = np.float32(adv_sample_g1_0)
adv_sample_g1_0 = np.reshape(adv_sample_g1_0, (9000,1))

adv_sample_g0_0001 = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_g0_0001.csv', delimiter=',')
adv_sample_g0_0001 = np.float32(adv_sample_g0_0001)
adv_sample_g0_0001 = np.reshape(adv_sample_g0_0001, (9000,1))

adv_sample_diff = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_diffvar_10000.csv', delimiter=',')
adv_sample_diff = np.float32(adv_sample_diff)
adv_sample_diff = np.reshape(adv_sample_diff, (9000,1))

adv_sample_l2 = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_l2.csv', delimiter=',')
adv_sample_l2 = np.float32(adv_sample_l2)
adv_sample_l2 = np.reshape(adv_sample_l2, (9000,1))


#ymax = 7
#ymin = -3

ymax = 2
ymin = -2
fig, axs = plt.subplots(3, 1, figsize=(80,40), sharex=True)

#axs[0].plot(X_test[0,0:4000,:], color='black')
axs[0].plot(adv_sample_l2[0:4000,:]-X_test[0,0:4000,:])
#axs[0].plot(adv_sample_g1_0[0:4000,:], color='lightblue', label='adv g1.0 data')
#axs[0].plot(adv_sample_g0_0001[0:4000,:], color='lightblue', label='adv g0.001 data')
#axs[0].plot(adv_sample[0,0:4000,:], color='blue', label='adv l2 data')
#axs[0].plot(X_test[0,:])
#axs[0].set_title('Original signal O')
axs[0].set_title('Perturbation(l2)')
axs[0].set_ylim([ymin, ymax])
#axs[0].set_xlabel('index')
axs[0].set_ylabel('signal value')
axs[0].legend()
#axs[1].plot(adv_sample[0,0:4000,:])
##axs[1].plot(adv_sample[0,:])
#axs[1].set_title('Adversarial signal {}'.format(ann_label))
#axs[1].set_ylim([ymin, ymax])
#axs[1].set_xlabel('index')
#axs[1].set_ylabel('signal value')

#axs[1].plot(adv_sample_g1_0[0:4000,:]-X_test[0,0:4000,:], color='lightblue')
#axs[1].plot(adv_sample_l2[0:4000,:], color='blue', label='adv l2 data')
axs[1].plot(adv_sample_g0_0001[0:4000,:]-X_test[0,0:4000,:], color='blue', label='adv l2 data')
#axs[1].plot(adv_sample[0,0:4000,:]-X_test[0,0:4000,:], color='lightblue')
#axs[1].plot(adv_sample_g0_0001[0:4000,:]-X_test[0,0:4000,:], color='lightgreen')
#axs[1].set_title('Adv signal N(l2)')
axs[1].set_title('Perturbation(sdtw)')
axs[1].set_ylim([ymin, ymax])
#axs[1].set_xlabel('index')
axs[1].set_ylabel('signal value')

#axs[2].plot(adv_sample_g0_0001[0:4000,:], color='blue')
axs[2].plot(adv_sample_diff[0:4000,:]-X_test[0,0:4000,:], color='blue')
#axs[2].set_title('Adv signal N(sdtw)')
axs[2].set_title('Perturbation(smooth metric)')
axs[2].set_ylim([ymin, ymax])
#axs[2].set_xlabel('index')
axs[2].set_ylabel('signal value')


#axs[3].plot(adv_sample_diff[0:4000,:], color='blue')
#axs[3].set_title('Adv signal N(smooth metric)')
#axs[3].set_ylim([ymin, ymax])
#axs[3].set_xlabel('index')
#axs[3].set_ylabel('signal value')


a = np.zeros(5)
b = np.array([-1, 1, -1, 1, -1], dtype=np.float64)
l2_ab = np.sum(np.square(b-a))

c = np.array([-np.sqrt(0.5)*2, -np.sqrt(0.5), 0, np.sqrt(0.5), np.sqrt(0.5)*2], dtype=np.float64)
l2_ac = np.sum(np.square(c-a))

from mysoftdtw_c_wd import mysoftdtw
import tensorflow as tf
tmp_a = a.reshape((1, 5, 1))
tmp_a = np.float32(tmp_a)
tmp_b = b.reshape((1, 5, 1))
tmp_b = np.float32(tmp_b)
tmp_c = c.reshape((1, 5, 1))
tmp_c = np.float32(tmp_c)
A = tf.constant(tmp_a)
B = tf.constant(tmp_b)
C = tf.constant(tmp_c)
gamma = tf.constant(0.1, dtype=tf.float32)
Z_ab = mysoftdtw(A, B, gamma)
Z_ac = mysoftdtw(A, C, gamma)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
print(Z_ab.eval(session=sess))
print(Z_ac.eval(session=sess))

diff_ab = np.var(np.diff(c-a))
diff_ac = np.var(np.diff(b-a))

plt.plot(a)
plt.plot(b)