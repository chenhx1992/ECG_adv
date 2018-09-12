#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:58:06 2018

@author: chenhx1992
"""

import numpy as np
import csv
import scipy.io
import glob
from six.moves import xrange

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
from tensorflow.python.platform import flags
import logging


from cleverhans.attacks import SaliencyMapMethod
#from cleverhans.loss import LossCrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans import utils
from distutils.version import LooseVersion

# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

viz_enabled=True
batch_size=1
nb_classes=4
source_samples=10
img_rows = 9000
img_cols = 1
channels = 1
    
report = AccuracyReport()


# loading model
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

# From cleverhans
wrap = KerasModelWrapper(model)
tf.set_random_seed(1234) # Set TF random seed to improve reproducibility
sess = keras.backend.get_session()
set_log_level(logging.DEBUG)

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
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
    x = np.expand_dims(x, axis=2)  # required by Keras
    del tmp
    
    return x

preds = model(x)

#loss = LossCrossEntropy(model, smoothing=0.1)


print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

# Keep track of success (adversarial example classified in target)
results = np.zeros((nb_classes, source_samples), dtype='i')

# Rate of perturbed features for each test set example and target class
perturbations = np.zeros((nb_classes, source_samples), dtype='f')

# Initialize our array for grid visualization
grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
grid_viz_data = np.zeros(grid_shape, dtype='f')

# Instantiate a SaliencyMapMethod attack object
jsma = SaliencyMapMethod(wrap, sess=sess)
jsma_params = {'theta': 1., 'gamma': 0.1,
               'clip_min': 0., 'clip_max': 1.,
               'y_target': None}

figure = None

## Loading time serie signals
# Loop over the samples we want to perturb into adversarial examples
for id in range(source_samples):
    id = id+1
    count = id-1
    record = "A{:05d}".format(id)
    local_filename = "./training_raw/"+record
    # Loading
    mat_data = scipy.io.loadmat(local_filename)
    print('Loading record {}'.format(record))    
    #    data = mat_data['val'].squeeze()
    data = mat_data['val']
    
    data = preprocess(data, WINDOW_SIZE)
    
    ground_truth_label = csvfile[count][1]
    ground_truth = classes.index(ground_truth_label)
    print('Ground truth:{}'.format(ground_truth))
    X_test = data
    #X_test = np.expand_dims(X_test, -1)
    Y_test = np.zeros((1, 1))
    Y_test[0,0] = ground_truth
    Y_test = utils.to_categorical(Y_test, num_classes=4)
    
#    feed_dict = {x: X_test, y: Y_test}
    
    print('--------------------------------------')
    print('Attacking input {}'.format(id))
    sample =np.float32(X_test)

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = ground_truth
    target_classes = other_classes(nb_classes, current_class)

    # For the grid visualization, keep original images along the diagonal
    grid_viz_data[current_class, current_class, :, :, :] = np.reshape(sample, (img_rows, img_cols, channels))

    # Loop over all target classes
    for target in target_classes:
        print('Generating adv. example for target class %i' % target)

        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target] = 1
        one_hot_target = np.float32(one_hot_target)
        jsma_params['y_target'] = one_hot_target
#        adv_x = jsma.generate(x, **jsma_params)
        adv_x = jsma.generate_np(sample, **jsma_params)
        
        # Check if success was achieved
        res = int(model_argmax(sess, x, preds, adv_x) == target)

        # Computer number of modified features
        adv_x_reshape = adv_x.reshape(-1)
        test_in_reshape = sample.reshape(-1)
        nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
        percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

        # Display the original and adversarial images side-by-side
        if viz_enabled:
            figure = pair_visual(
                np.reshape(sample, (img_rows, img_cols, channels)),
                np.reshape(adv_x, (img_rows, img_cols, channels)), figure)

        # Add our adversarial example to our grid data
        grid_viz_data[target, current_class, :, :, :] = np.reshape(
            adv_x, (img_rows, img_cols, channels))

        # Update the arrays for later analysis
        results[target, count] = res
        perturbations[target, count] = percent_perturb

print('--------------------------------------')

# Compute the number of adversarial examples that were successfully found
nb_targets_tried = ((nb_classes - 1) * source_samples)
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
    import matplotlib.pyplot as plt
    plt.close(figure)
    _ = grid_visual(grid_viz_data)



