#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:07:06 2018

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

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans import utils
from distutils.version import LooseVersion


#from cleverhans.attacks import CarliniWagnerL2  

import csv
import scipy.io
import glob
import numpy as np
#import matplotlib.pyplot as plt

# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

keras.layers.core.K.set_learning_phase(0)
# loading model
sess = tf.Session()
K.set_session(sess)

print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')
#model = load_model('weights-best_k0_r0.hdf5')

wrap = KerasModelWrapper(model, nb_classes=4)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

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

preds = model(x)

## Loading time serie signals
id = 5
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

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))

X_test=np.float32(data)
Y_test = np.zeros((1, 1))
Y_test[0,0] = ground_truth
Y_test = utils.to_categorical(Y_test, num_classes=4)

target_a = np.array([0, 1, 0, 0]).reshape(1,4)
target_a = np.float32(target_a)

X_test_1 = zero_mean(X_test)
prob = model.predict(X_test_1)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

## myattacks CWL2
from myattacks_sdtw import CarliniWagnerL2
import time
cwl2 = CarliniWagnerL2(wrap, sess=sess)
cwl2_params = {'y_target': target_a}
adv_x = cwl2.generate(x, **cwl2_params)
adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
#preds_adv = model(adv_x)
feed_dict = {x: X_test}
#adv_sample = sess.run(adv_x, feed_dict=feed_dict)
start_time = time.time()  
adv_sample = adv_x.eval(feed_dict=feed_dict, session = sess)
print("--- %s seconds ---" % (time.time() - start_time))
#adv_sample = cwl2.generate_np(X_test, **cwl2_params)


prob = model.predict(adv_sample)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

adv_sample_1 = zero_mean(adv_sample)
prob = model.predict(adv_sample_1)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

X_test_1 = zero_mean(X_test)

np.savetxt('./result_6755/R6755_0_1_1_diff_1.csv', adv_sample[0,:], delimiter=",")

from numpy import genfromtxt
adv_sample_g1_0 = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_g1_0.csv', delimiter=',')
adv_sample_g1_0 = np.float32(adv_sample_g1_0)
adv_sample_g1_0 = np.reshape(adv_sample_g1_0, (9000,1))

adv_sample_g0_0001 = genfromtxt('./result_6755/R' + str(id) + '_0_1_1_g0_0001.csv', delimiter=',')
adv_sample_g0_0001 = np.float32(adv_sample_g0_0001)
adv_sample_g0_0001 = np.reshape(adv_sample_g0_0001, (9000,1))

import matplotlib.pyplot as plt

ymax = np.max(adv_sample)+0.5
ymin = np.min(adv_sample)-0.5

dist = np.var(np.diff(adv_sample[0,:]-X_test[0,:], axis=0), axis=0)

fig, axs = plt.subplots(2, 1, figsize=(80,40), sharex=True)

axs[0].plot(X_test[0,0:4000,:], color='black')
#axs[0].plot(X_test_1[0,0:4000,:], color='black')
#axs[0].plot(adv_sample_g1_0[0:4000,:], color='lightblue', label='adv g1.0 data')
#axs[0].plot(adv_sample_g0_0001[0:4000,:], color='lightblue', label='adv g0.001 data')
#axs[0].plot(adv_sample[0,0:4000,:], color='blue', label='adv l2 data')
#axs[0].plot(X_test[0,:])
axs[0].set_title('Original signal {}'.format(ground_truth_label))
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
axs[1].plot(adv_sample_1[0,0:4000,:], color='blue', label='adv l2 data')
axs[1].plot(adv_sample_1[0,0:4000,:]-X_test[0,0:4000,:], color='green', label='adv l2 data')
#axs[1].plot(adv_sample[0,0:4000,:]-X_test[0,0:4000,:], color='lightblue')
#axs[1].plot(adv_sample_g0_0001[0:4000,:]-X_test[0,0:4000,:], color='lightgreen')
axs[1].set_title('Adv signal N')
axs[1].set_ylim([ymin, ymax])
axs[1].set_xlabel('index')
axs[1].set_ylabel('signal value')

#plt.plot(adv_sample[0,:]-X_test[0,:])
#plt.ylim([-0.1, 0.1])

#fig.savefig('p9.png',dpi=fig.dpi)