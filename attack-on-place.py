#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint
from random import uniform
import csv
import scipy.io
import math
import glob
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


def get_gradient_signs(model, original_array, targid):
    target_idx = targid #model.predict(original_array).argmin()
    target = to_categorical(target_idx, 4)
    target_variable = K.variable(target)
    num_samples = 30
    grad_values = 0
    for i in range(num_samples):
        print(i)
        shift_ind = randint(0, 3000)
        ecg_shift = np.roll(original_array, -shift_ind)
        layer_name = 'dense_{}'.format(1)
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer_output = layer_dict[layer_name].output
        loss = metrics.categorical_crossentropy(layer_output, target_variable)
        #average_loss = average_loss + loss / num_samples
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = grad_values + get_grad_values([ecg_shift])[0]/num_samples
    #grad_signs = np.sign(grad_values)
    grad_value = grad_values/np.amax(grad_values)
    return grad_value


def pertubate_image(preprocessed_array, perturbation):

    modified_array = preprocessed_array + perturbation
    deprocess_array = np.clip(modified_array, -20, 20).astype(np.float16)
    return deprocess_array


def generate_adversarial_example(pertubation_model, original_array, epsilon, targid):

    gradient_signs = get_gradient_signs(pertubation_model, original_array, targid)
    perturbation = gradient_signs * epsilon
    modified_data = pertubate_image(original_array, perturbation)
    return modified_data, perturbation


## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))

# Load and apply model
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))

# Main loop 
consist = 0
id = 6
count = id-1
record = "A{:05d}".format(id)
local_filename = "./training_raw/"+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']

x = preprocess(data, WINDOW_SIZE)

prob_x, ann_x = predict_data(model, x)
ann_original = ann_x

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print("Record {} ground truth: {}".format(record, ground_truth_label))

# From chenyu
epsilon = 1
max_epoches = 1
targid = 2
epoch = 0
succ = 0
max_case = 3000
max_iter = 2
iter = 0
curr_data = x
while  (iter < max_iter):
    iter = iter + 1
    modified_data, perturbation = generate_adversarial_example(model, curr_data, epsilon, targid)
    curr_data = modified_data
    if (iter == 1):
        perturbation_all = perturbation
    else:
        perturbation_all = perturbation_all + perturbation


perturbation = perturbation_all

plt.figure()
plt.plot(perturbation_all.squeeze())
plt.show()

max_case = 3000
origin_data = x
for l in range(0, max_case):
    rand_shift = randint(0, 3000)
    temp_perturbation = np.roll(perturbation, -rand_shift)
    modified_data = pertubate_image(origin_data, temp_perturbation * uniform(0.8,1.2))
    prob = model.predict(modified_data)
    ann = np.argmax(prob)

    print("probability:",prob)
    print("class:{}", classes[ann_x])
    if ann!=ann_original:
        succ = succ + 1

print("success",succ)
print("success rate:",succ/max_case)


