import numpy as np
import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
from random import randrange
import csv
import glob
import scipy.io
import math


def preprocess(x, maxlen):
    x = np.nan_to_num(x)
    x = x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    tmp = np.zeros((1, maxlen))
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
    x = np.expand_dims(x, axis=2)  # required by Keras
    del tmp
    return x
def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

# parameters
dataDir = '../training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

print("Loading ground truth file")
csvfile = list(csv.reader(open('../REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

id_1 = 51
id_2 = 5
target = 1
count = id_1-1
record_1 = "A{:05d}".format(id_1)
record_2 = "A{:05d}".format(id_1)

# Loading record 1
local_filename = dataDir+record_1
mat_data = scipy.io.loadmat(local_filename)
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)
X_test_1 = np.float32(data)


# Loading record 2
local_filename = dataDir+record_2
mat_data = scipy.io.loadmat(local_filename)
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)
X_test_2 = np.float32(data)

#loading ground truth
ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))



def op_concate(x, p):
    data_len = 9000
    x1 = x[:, 0:p, :]
    x2 = x[:, p:data_len, :]
    return np.append(x2, x1, axis=1)

print("Loading model")
model = load_model('../ResNet_30s_34lay_16conv.hdf5')


attack_success = np.zeros(4)

test_all = zero_mean(X_test_1 + X_test_2)
for i in range(99):
    p = randrange(0, 9000)
    test_all = np.append(test_all, zero_mean(op_concate(X_test_2, p) + X_test_1), axis=0)

prob = model.predict(test_all)
ind = np.argmax(prob, axis=1)
for _,it in enumerate(ind):
    attack_success[it] = attack_success[it] + 1
print(test_all.shape)
#print("correct:", correct)
print("attack success times:", attack_success)

import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(perturb[0,:,0])
#plt.show(block=False)

#adv_sample = op_concate(perturb,perturb_window,False) + X_test
#plt.figure()
#plt.plot(adv_sample[0,1000:2000,0])
#plt.show(block=False)

#plt.figure()
#plt.plot(X_test[0,1000:2000,0])
#plt.show(block=False)
