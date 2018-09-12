#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans import utils
from distutils.version import LooseVersion

import csv
import scipy.io
import glob
import numpy as np

# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

# loading model
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

# From cleverhans
wrap = KerasModelWrapper(model)
#sess = tf.Session()
#keras.backend.set_session(sess)
sess = keras.backend.get_session()
#init_op = tf.initialize_all_variables()
#sess.run(init_op)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

modeltf = tf.keras.models.load_model('ResNet_30s_34lay_16conv.hdf5')

#plot_model(model, to_file='model.png')

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open(dataDir+'REFERENCE-v3.csv')))

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

fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate(x, **fgsm_params)
# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv = model(adv_x)
# Define accuracy symbolically
if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
    original_preds = tf.argmax(preds, axis=-1)
    adv_preds = tf.argmax(preds_adv, axis=-1)
else:
    original_preds = tf.argmax(preds, axis=tf.rank(preds) - 1)
    adv_preds = tf.argmax(preds_adv, axis=tf.rank(preds_adv) - 1)

files = sorted(glob.glob(dataDir+"*.mat"))

result_sheet = np.zeros((8528, 3))

## Loading time serie signals
for id in range(8528):
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
    
    feed_dict = {x: X_test, y: Y_test}
    original_res = original_preds.eval(feed_dict, session=sess)
    adv_res = adv_preds.eval(feed_dict, session=sess)
    print('Original result: {}'.format(original_res[0]))
    print('Adv result: {}'.format(adv_res[0]))
    result_sheet[count, 0] = ground_truth
    result_sheet[count, 1] = original_res[0]
    result_sheet[count, 2] = adv_res[0]


import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
 # Compute confusion matrix
cnf_matrix = confusion_matrix(result_sheet[:,1], result_sheet[:,2])
np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix, without normalization')
    
#    # Evaluate the accuracy of model on examples
#    eval_par = {'batch_size': 1}
#    acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_par)
#    print('Test accuracy on examples: %0.4f' % acc)
#    
#    prob = model.predict(X_test)
#    ann = np.argmax(prob)
#    print(ann)
#    
#    # Evaluate the accuracy of model on adversarial examples
#    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
#    print('Test accuracy on adversarial examples: %0.4f\n' % acc)

