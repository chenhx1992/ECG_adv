#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:39:19 2018

@author: chenhx1992
"""

## General library
import scipy.io
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## keras and tensorflow library
import tensorflow as tf
import keras
from keras import backend
from keras.models import load_model
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras import metrics
from tensorflow.python.platform import flags
import logging

## cleverhans library
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans import utils
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from distutils.version import LooseVersion


## helper function
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


def plot_x_advx_per(x, adv_x, y, adv_y):
    ymax = np.max(adv_x)+0.5
    ymin = np.min(adv_x)-0.5
    
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    
    axs[0].plot(x[0,:])
    axs[0].set_title('Original signal {}'.format(y))
    axs[0].set_ylim([ymin, ymax])
    axs[0].set_xlabel('index')
    axs[0].set_ylabel('signal value')
    
    axs[1].plot(adv_x[0,:])
    axs[1].set_title('Adversarial signal {}'.format(adv_y))
    axs[1].set_ylim([ymin, ymax])
    axs[1].set_xlabel('index')
    axs[1].set_ylabel('signal value')
    
    
    axs[2].plot(adv_x[0,:]-x[0,:])
    axs[2].set_title('perturbations')
    axs[2].set_ylim([ymin, ymax])
    axs[2].set_xlabel('index')
    axs[2].set_ylabel('signal value')

# parameters
dataDir = './training_raw/'
files = sorted(glob.glob(dataDir+"*.mat"))
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']
nb_class = 4
rows = 9000
cols = 1
channels = 1

# load Correct Prediction file
print("Loading correct prediction file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))
df=pd.read_csv('./DataAnalysis/prediction_correct.csv', sep=',',header=None)
tmp = np.expand_dims(df.values, axis=2)
datafile_index = np.append(tmp[:, 0], tmp[:, 8], axis=1)
source_samples = len(datafile_index)

# Model loading and process
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')
wrap = KerasModelWrapper(model, nb_classes = nb_class)
sess = keras.backend.get_session()

set_log_level(logging.DEBUG)

#tf.set_random_seed(1234) # Set TF random seed to improve reproducibility
x = tf.placeholder(tf.float32, shape=(None, rows, cols))
y = tf.placeholder(tf.float32, shape=(None, nb_class))
preds = model(x)

## target tensor for source-target attack 
target_a = np.array([1, 0, 0, 0]).reshape(1,4)
target_a = np.float32(target_a)
#target_a = tf.convert_to_tensor(target_a, np.float32)

target_n = np.array([0, 1, 0, 0]).reshape(1,4)
target_n = np.float32(target_n)
#target_n = tf.convert_to_tensor(target_n, np.float32)

target_o = np.array([0, 0, 1, 0]).reshape(1,4)
target_o = np.float32(target_o)
#target_o = tf.convert_to_tensor(target_o, np.float32)

target_s = np.array([0, 0, 0, 1]).reshape(1,4)
target_s = np.float32(target_s)
#target_s = tf.convert_to_tensor(target_s, np.float32)


## Result and visualization
# Keep track of success (adversarial example classified in target)
results = np.zeros((nb_class, source_samples), dtype='i')
report = AccuracyReport()
# Rate of perturbed features for each test set example and target class
perturbations = np.zeros((nb_class, source_samples), dtype='f')

# Initialize our array for grid visualization
grid_shape = (nb_class, nb_class, rows, cols, channels)
grid_viz_data = np.zeros(grid_shape, dtype='f')
viz_enabled=True
figure = None

## Defile Attack method parameters
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 20, 'ord': 2, 'y_target': None}

bim = BasicIterativeMethod(wrap, sess=sess)
bim_params = {'eps': 0.2, 'eps_iter': 0.05, 'nb_iter': 10, 'ord': 2, 'y_target': None}

#source_samples = 10
type_a_num = 0
type_n_num = 0
type_o_num = 0
type_s_num = 0
result_log = np.zeros((source_samples, 3))
for i in range(source_samples):
    id = int(datafile_index[i,1])
    ground_truth = datafile_index[i,0]
    ground_truth_label = classes[int(ground_truth)]
    
    # source target access 
    if ground_truth == 0:
        target = 1
        one_hot_target = target_n
        type_a_num +=1
    if ground_truth == 1:
#        continue
        target = 2
        one_hot_target = target_o
        type_n_num +=1
    if ground_truth == 2:
#        continue
        target = 1
        one_hot_target = target_n
        type_o_num +=1
    if ground_truth == 3:
#        continue
        target = 1
        one_hot_target = target_n
        type_s_num +=1
        
    record = "A{:05d}".format(id)
    local_filename = dataDir+record
    # Loading
    mat_data = scipy.io.loadmat(local_filename)
    print('Loading record {}'.format(record))
    print('Ground truth:{}({})'.format(ground_truth, ground_truth_label))    
    #    data = mat_data['val'].squeeze()
    data = mat_data['val']
    
    data = preprocess(data, WINDOW_SIZE)
    sample = np.float32(data)
    
    # For the grid visualization, keep original images along the diagonal
#    grid_viz_data[ground_truth, ground_truth, :, :, :] = np.reshape(sample, (rows, cols, channels))
    
    print('Generating adv. example for target class %i' % target)
    # This call runs specific attack approach
    fgsm_params['y_target'] = one_hot_target
#        adv_x = jsma.generate(x, **jsma_params)
    adv_sample = fgsm.generate_np(sample, **fgsm_params)
    
#    bim_params['y_target'] = one_hot_target
#    adv_sample = bim.generate_np(sample, **bim_params)
    
    prob = model.predict(adv_sample)
    adv_y = np.argmax(prob)
    adv_label = classes[int(adv_y)]
    
    result_log[i, 0] = id
    result_log[i, 1] = ground_truth
    result_log[i, 2] = adv_y
    
    if (i < 10):
        plot_x_advx_per(sample, adv_sample, ground_truth_label, adv_label)     
    
#    # Check if success was achieved
#    res = int(model_argmax(sess, x, preds, adv_x) == target)
#    
#    # Computer number of modified features
#    adv_x_reshape = adv_x.reshape(-1)
#    test_in_reshape = sample.reshape(-1)
#    nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
#    percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]
#
#    # Display the original and adversarial images side-by-side
#    if viz_enabled:
#        figure = pair_visual(np.reshape(sample, (rows, cols, channels)),
#            np.reshape(adv_x, (rows, cols, channels)), figure)
#
#    # Add our adversarial example to our grid data
#    grid_viz_data[target, ground_truth, :, :, :] = np.reshape(adv_x, (rows, cols, channels))
#
#    # Update the arrays for later analysis
#    results[target, id-1] = res
#    perturbations[target, id-1] = percent_perturb
    
confusion_matrix = np.zeros((4, 4))
for i in range(source_samples):
    if int(result_log[i, 0]) > 0:
        confusion_matrix[int(result_log[i, 1]), int(result_log[i, 2])] +=1
plt.matshow(confusion_matrix)
    
format = '%i,%i,%i'
np.savetxt("./DataAnalysis/attack_fgsm_eps10_ord2_target.csv", result_log, fmt= format, delimiter=",")
    
print('--------------------------------------')

# Compute the number of adversarial examples that were successfully found
nb_targets_tried = source_samples
succ_rate = float(np.sum(results)) / nb_targets_tried
print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
report.clean_train_adv_eval = 1. - succ_rate

# Compute the average distortion introduced by the algorithm
percent_perturbed = np.mean(perturbations)
print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

# Compute the average distortion introduced for successful samples only
percent_perturb_succ = np.mean(perturbations * (results == 1))
print('Avg. rate of perturbed features for successful '
      'adversarial examples {0:.4f}'.format(percent_perturb_succ))

# Close TF session
#sess.close()

# Finally, block & display a grid of all the adversarial examples
if viz_enabled:
    plt.close(figure)
    _ = grid_visual(grid_viz_data)
    


