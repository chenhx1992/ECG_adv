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

wrap = KerasModelWrapper(model)

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

preds = model(x)

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
print(data.shape)
data = preprocess(data, WINDOW_SIZE)

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))

X_test=np.float32(data)
Y_test = np.zeros((1, 1))
Y_test[0,0] = ground_truth
Y_test = utils.to_categorical(Y_test, num_classes=4)

# zero signal test
X_test_0 = np.zeros((1, 9000, 1), dtype=np.float32)
prob = model.predict(X_test_0)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

target_a = np.array([1, 0, 0, 0]).reshape(1,4)
target_a = np.float32(target_a)

## myattacks CWL2
from myattacks import CarliniWagnerL2  
cwl2 = CarliniWagnerL2(wrap, sess=sess)
cwl2_params = {'y_target': target_a}
adv_x = cwl2.generate(x, **cwl2_params)
adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
#preds_adv = model(adv_x)
feed_dict = {x: X_test}
feed_dict = {x: X_test_0}
#adv_sample = sess.run(adv_x, feed_dict=feed_dict)
adv_sample = adv_x.eval(feed_dict=feed_dict, session = sess)

#adv_sample = cwl2.generate_np(X_test, **cwl2_params)

prob = model.predict(adv_sample)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

np.savetxt('./result_zero/zero_tarA_l2.out', adv_sample[0,:], delimiter=",")
from numpy import genfromtxt
perturb = genfromtxt('./result_zero/zero_tarA.out', delimiter=',')

#plt.plot(X_test_0[0,:])
import matplotlib.pyplot as plt
plt.plot(adv_sample[0,:])
plt.xlabel('index')
plt.ylabel('signal value')
plt.ylim([-6, 6])
#ymax = np.max(adv_sample)+0.5
#ymin = np.min(adv_sample)-0.5
#
#fig, axs = plt.subplots(2, 1, figsize=(50,40))
#
#axs[0].plot(X_test[0,0:4000,:])
##axs[0].plot(X_test[0,:])
#axs[0].set_title('Original signal {}'.format(ground_truth_label))
#axs[0].set_ylim([ymin, ymax])
##axs[0].set_xlabel('index')
#axs[0].set_ylabel('signal value')
#
#axs[1].plot(adv_sample[0,0:4000,:])
##axs[1].plot(adv_sample[0,:])
#axs[1].set_title('Adversarial signal {}'.format(ann_label))
#axs[1].set_ylim([ymin, ymax])
#axs[1].set_xlabel('index')
#axs[1].set_ylabel('signal value')
'''
axs[2].plot(adv_sample[0,:]-X_test[0,:])
axs[2].set_title('perturbations')
axs[2].set_ylim([ymin, ymax])
axs[2].set_xlabel('index')
axs[2].set_ylabel('signal value')
'''
#fig.savefig('p9.png',dpi=fig.dpi)