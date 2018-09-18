#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 01:22:12 2018

@author: chenhx1992
"""

import scipy.io
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras import metrics

import tensorflow as tf

# Parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

## funtion 
def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    
    tmp = np.zeros((1, maxlen))
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
    x = np.expand_dims(x, axis=2)  # required by Keras
    del tmp
    
    return x
    
def concact_data(x):
    data_len = len(x[0,:])
    print("Data length:",data_len)
    x1 = [x[0,0:data_len // 2]]
    x2 = [x[0,data_len // 2:]]
    res = np.append(x2, x1, axis=1)
    print(res.shape)
    return res

def predict_data(model, x):
    prob = model.predict(x)
    ann = np.argmax(prob)
    return prob, ann
    
## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))

# Load and apply model
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))

# Main loop 
prediction = np.zeros((len(files),8))
count = 0
correct_consist = 0
correct_whole = 0
consist = 0
correct_30s = 0
correct_consist_30s = 0 

for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
    data = mat_data['val']
    
    x = preprocess(data, WINDOW_SIZE)
    
    x1 = concact_data(data)
    x1 = preprocess(x1, WINDOW_SIZE)
#    x2 = preprocess(x2, WINDOW_SIZE)
    
    
    print("Applying model ..") 
    ground_truth_label = csvfile[count][1]
    ground_truth = classes.index(ground_truth_label)
    prob_x, ann_x = predict_data(model, x)
    prob_x1, ann_x1 = predict_data(model, x1)
#    prob_x2, ann_x2 = predict_data(model, x2)
    
    print("Record {} ground truth: {}".format(record, ground_truth_label))
    print("Record {} classified as {} with {:3.1f}% certainty".format(record, classes[ann_x], 100*prob_x[0,ann_x]))
    print("Record {} first half classified as {} with {:3.1f}% certainty".format(record, classes[ann_x1], 100*prob_x1[0,ann_x1]))
#    print("Record {} second half classified as {} with {:3.1f}% certainty".format(record, classes[ann_x2], 100*prob_x2[0,ann_x2]))
    
    prediction[count,0] = ground_truth
    prediction[count,1] = ann_x
    prediction[count,2] = prob_x[0,ann_x]
    prediction[count,3] = ann_x1
    prediction[count,4] = prob_x1[0,ann_x1]
#    prediction[count,5] = ann_x2
#    prediction[count,6] = prob_x2[0,ann_x2]
#    prediction[count,5] = len(data[0,:])/300.0
    prediction[count,5] = len(data[0,:])
    
    if (ground_truth == ann_x):
        correct_whole += 1
        if (len(data[0,:]) == 9000):
            correct_30s +=1

    if (ann_x == ann_x1):
        consist += 1
        
    if (ground_truth == ann_x) and (ann_x == ann_x1):
        correct_consist += 1
        if (len(data[0,:]) == 9000):
            correct_consist_30s +=1
    count += 1
    
#    if count == 100:
#        break


print("Correct:{}, total:{}, percent:{}".format(correct_whole, count, correct_whole/(count)))
print("Consist:{}, total:{}, percent:{}".format(consist, count, consist/(count)))
print("Correct_consist:{}, total:{}, percent:{}".format(correct_consist, count, correct_consist/(count)))
print("Correct_consist_30s:{}, total_correct_30s:{}, percent:{}".format(correct_consist_30s, correct_30s, correct_consist_30s/(correct_30s)))

##append file index
#new_prediction = np.zeros((len(files),6))
#new_prediction[:, 0:5] = prediction[:, 0:5]
#new_prediction[:, 5] = np.arange(8528)+1
#
#Select correct and correct_consist prediction
#cond_x_x1 = np.equal(new_prediction[:,1], new_prediction[:,3])
##cond_x_x2 = np.equal(new_prediction[:,1], new_prediction[:,5]) 
#cond_x_gt = np.equal(new_prediction[:,1], new_prediction[:,0]) 
#cond_30_second = np.equal(int(new_prediction[:,5]), 9000) 
##cond_consist = np.logical_and(cond_x_x1, cond_x_x2)
#cond_correct_consist = np.logical_and(cond_x_x1, cond_x_gt)
#cond_correct_consist_30 = np.logical_and(cond_correct_consist, cond_30_second)
#cond_correct_30 = np.logical_and(cond_x_gt, cond_30_second)
#
#correct_consist_prediction = new_prediction[cond_correct_consist]
#correct_prediction = new_prediction[cond_x_gt]
#
## save prediction to csv files
#format = '%i,%i,%.5f,%i,%.5f,%i,%.5f,%.2f,%i'
#np.savetxt("./DataAnalysis/prediction_all.csv", new_prediction, fmt= format, delimiter=",")
#np.savetxt("./DataAnalysis/prediction_correct.csv", correct_prediction, fmt= format, delimiter=",")
#np.savetxt("./DataAnalysis/prediction_correct_consist.csv", correct_consist_prediction, fmt= format, delimiter=",")
#
## check each type percentage in correct_consist prediction
#type_a = correct_consist_prediction[correct_consist_prediction[:,0] == 0]
#type_n = correct_consist_prediction[correct_consist_prediction[:,0] == 1]
#type_o = correct_consist_prediction[correct_consist_prediction[:,0] == 2]
#type_s = correct_consist_prediction[correct_consist_prediction[:,0] == 3]
#
#type_a_all = new_prediction[new_prediction[:,0] == 0]
#type_n_all = new_prediction[new_prediction[:,0] == 1]
#type_o_all = new_prediction[new_prediction[:,0] == 2]
#type_s_all = new_prediction[new_prediction[:,0] == 3]