import numpy as np
import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
from numpy import genfromtxt
import csv
import glob
import scipy.io


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

# parameters
dataDir = '../training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

print("Loading ground truth file")
csvfile = list(csv.reader(open('../REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

id = 9
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


perturb = genfromtxt('../output/EOT_t30_f1_l2_A10T2.out', delimiter=',')

perturb = np.expand_dims(perturb, axis=0)
perturb = np.expand_dims(perturb, axis=2)

def op_concate(x):
    data_len = 9000
    p = np.random.randint(data_len)
    x1 = [x[0, 0:p]]
    x2 = [x[0, p:]]
    return np.append(x2, x1, axis=1)

print("Loading model")
model = load_model('../ResNet_30s_34lay_16conv.hdf5')


attack_success = 0

for _ in range(100):
    #new_X_test = op_concate(X_test)
    #prob_ori = model.predict(new_X_test)
    prob_att = model.predict(op_concate(perturb)+X_test)
    #if np.argmax(prob_ori) == ground_truth:
        #correct = correct + 1
    if np.argmax(prob_att) != ground_truth:
        attack_success = attack_success + 1
#print("correct:", correct)
print("attack success times:", attack_success)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(perturb[0,1000:2000,0])
plt.show()

adv_sample = perturb+X_test
plt.figure()
plt.plot(adv_sample[0,1000:2000,0])
plt.show()

plt.figure()
plt.plot(X_test[0,1000:2000,0])
plt.show()