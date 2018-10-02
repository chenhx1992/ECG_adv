#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:52:21 2018

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
import time
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
model = load_model('./ResNet_30s_34lay_16conv.hdf5')
#model = load_model('weights-best_k0_r0.hdf5')

wrap = KerasModelWrapper(model, nb_classes=4)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

preds = model(x)

## Loading time serie signals

# zero signal test
X_test_0 = np.zeros((1, 9000, 1), dtype=np.float32)

prob = model.predict(X_test_0)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

target_a = np.array([1, 0, 0, 0]).reshape(1,4)
target_a = np.float32(target_a)

## myattacks CWL2
from myattacks_sdtw import CarliniWagnerL2  
cwl2 = CarliniWagnerL2(wrap, sess=sess)
cwl2_params = {'y_target': target_a}
adv_x = cwl2.generate(x, **cwl2_params)
adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
#preds_adv = model(adv_x)
feed_dict = {x: X_test_0}
#adv_sample = sess.run(adv_x, feed_dict=feed_dict)

start_time = time.time()
adv_sample = adv_x.eval(feed_dict=feed_dict, session = sess)
#adv_sample = cwl2.generate_np(X_test, **cwl2_params)
print("--- %s seconds ---" % (time.time() - start_time))

prob = model.predict(adv_sample)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

np.savetxt('./result_zero/zero_tarA_wd300_gamma_00001.out', adv_sample[0,:], delimiter=",")

##from numpy import genfromtxt
##perturb = genfromtxt('./result_zero/zero_tarA.out', delimiter=',')
#
import matplotlib.pyplot as plt
#plt.plot(X_test_0[0,:])

plt.plot(adv_sample[0,:])
plt.xlabel('index')
plt.ylabel('signal value')
plt.ylim([-6, 6])
