#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:57:58 2018

@author: chenhx1992
"""

#### Module import
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
import csv
import scipy.io
import glob
import numpy as np
import sys
from numpy import genfromtxt

# Code snippet: accept arguments from command line
#print('Number of arguments:', len(sys.argv), 'arguments.')
#print('Argument List:', str(sys.argv))
#print(sys.argv[1])
#print(sys.argv[2])
#print(type(int(sys.argv[1])))
#print(type(int(sys.argv[2])))

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

#### Main Program

#--- parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

fid_from = int(sys.argv[1])
fid_to = int(sys.argv[2])
data_select = genfromtxt('data_select.csv', delimiter=',')

#--- loading model and prepare wrapper
keras.layers.core.K.set_learning_phase(0)
sess = tf.Session()
K.set_session(sess)
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')
#model = load_model('weights-best_k0_r0.hdf5')

wrap = KerasModelWrapper(model, nb_classes=4)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))
preds = model(x)

#--- load groundTruth File
print("Loading ground truth file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

#--- Attacker
from myattacks_sdtw import CarliniWagnerL2  
cwl2 = CarliniWagnerL2(wrap, sess=sess)

#--- loop on file including data_select[:,2] from fid_from-th row to fid_to-th row

eval_result = np.zeros((4*(fid_to-fid_from), 4)) # fid, ground_truth, target, adv_result

num = fid_from
while (num < fid_to):
    
    #--- Loading
    fid = int(data_select[num, 2]) 
    record = "A{:05d}".format(fid)
    local_filename = "./training_raw/"+record
    print('Loading record {}'.format(record))    
    mat_data = scipy.io.loadmat(local_filename)
    #data = mat_data['val'].squeeze()
    data = mat_data['val']
    print(data.shape)
    
    #--- Processing data
    data = preprocess(data, WINDOW_SIZE)
    X_test=np.float32(data)
    
    #--- Read the ground truth label, Change it to one-shot form
    ground_truth_label = csvfile[fid-1][1]
    ground_truth = classes.index(ground_truth_label)
    print('Ground truth:{}'.format(ground_truth))
    
    Y_test = np.zeros((1, 1))
    Y_test[0,0] = ground_truth
    Y_test = utils.to_categorical(Y_test, num_classes=4)
    
    #--- Prepare the target labels for targeted attack
    for i in range(4):
        if (i == ground_truth):
            continue
        
        target = np.zeros((1, 1))
        target[0,0] = i
        target = utils.to_categorical(target, num_classes=4)
        target = np.float32(target)
        
        #--- Attacking...
        cwl2_params = {'y_target': target}
        adv_x = cwl2.generate(x, **cwl2_params)
        adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
        feed_dict = {x: X_test}
        adv_sample = adv_x.eval(feed_dict=feed_dict, session=sess)
        
        #--- Attack result
        prob = model.predict(adv_sample)
        ann = np.argmax(prob)
#        ann_label = classes[ann]
        print('Adv result:{}'.format(ann))
        
        eval_result[4*num+i, 0] = fid
        eval_result[4*num+i, 1] = ground_truth
        eval_result[4*num+i, 2] = i
        eval_result[4*num+i, 3] = ann
        
        #--- Save adv_sample to file
        file_sample = './cwsdtw_eval/R' + str(fid)+ '_' + str(ground_truth) + '_' + str(i) + '_' + str(ann) + '.csv'
        np.savetxt(file_sample, adv_sample[0,:], delimiter=",")
        
    num = num+1
        
file_result = './cwsdtw_eval/res'+ '_' + str(fid_from) + '_' + str(fid_to) + '.csv'
np.savetxt(file_result, eval_result, delimiter=",")  
        
    
    