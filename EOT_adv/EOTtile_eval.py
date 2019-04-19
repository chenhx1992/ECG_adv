import numpy as np
import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
from numpy import genfromtxt
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

id = 4575
target = 1
perturb_window = 200
ensemble_size = 30
count = id-1
record = "A{:05d}".format(id)
local_filename = dataDir+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))
#    data = mat_data['val'].squeeze()
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)
X_test=np.float32(data)

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))

inputstr = '../output/EOTtile_w'+str(perturb_window)+'_e'+str(ensemble_size)+'_l2_A'+str(id)+'T'+str(target)+'.out'
print("input file: ", inputstr)
perturb = genfromtxt(inputstr, delimiter=',')
dist = np.linalg.norm(perturb)
print("distance:", dist)
perturb = np.expand_dims(perturb, axis=0)
perturb = np.expand_dims(perturb, axis=2)

def op_concate(x, w, p):
    data_len = 9000
    tile_times = math.ceil(data_len/w)
    x_tile = np.tile(x, (1, tile_times, 1))
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]
    return np.append(x2, x1, axis=1)

print("Loading model")
'''
model = load_model('../ResNet_30s_34lay_16conv.hdf5')


attack_success = np.zeros(4)
perturb_window = perturb.shape[1]

for i in range(perturb_window):
    prob_att = model.predict(zero_mean(op_concate(perturb, perturb_window, i)+X_test))
    ind = np.argmax(prob_att)
    attack_success[ind] = attack_success[ind] + 1
    print(prob_att)
    if ind != target:
        print(prob_att, "not success:", i)

#print("correct:", correct)
print("attack success times:", attack_success)
'''
dist = np.linalg.norm(X_test[0,:,0])
dist = dist * dist
print(dist)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(perturb[0,:,0])


adv_sample = op_concate(perturb,perturb_window,False) + X_test
plt.figure()
plt.plot(adv_sample[0,:,0])


plt.figure()
plt.plot(X_test[0,:,0])
plt.show()
