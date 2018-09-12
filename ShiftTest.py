#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    
def shift_data(x, shift):
    shifted_data = np.roll(x, -shift) 
    
    return shifted_data

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
prediction = np.zeros((len(files),6001))
consist = 0
id = 5
count = id-1
record = "A{:05d}".format(id)
local_filename = "./training_raw/"+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']

x = preprocess(data, WINDOW_SIZE)

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print("Record {} ground truth: {}".format(record, ground_truth_label))
prediction[count,0] = ground_truth

prob_x, ann_x = predict_data(model, x)
ann_x_original = ann_x

for i in range(3000):
    shifted_x = shift_data(x, i)

    print("Applying model ..") 
    prob_x, ann_x = predict_data(model, shifted_x)

    print("Record {} shift {} classified as {} with {:3.1f}% certainty".format(record, i, classes[ann_x], 100*prob_x[0,ann_x]))

    prediction[count,2*i+1] = ann_x
    prediction[count,2*i+2] = prob_x[0,ann_x]

    if (i != 0) and (ann_x == ann_x_original):
        consist += 1
    
print("Record {}, consist:{}, percent:{}".format(record, consist, consist/2999))


shift_result = prediction[count, 1::2] 

plt.figure()
plt.plot(np.arange(3000)/300, shift_result)
plt.show()

plt.figure()
plt.plot(np.arange(9000)/300, x.squeeze()[0:9000].T)
plt.show()

