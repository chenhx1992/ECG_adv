import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
import time
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import utils

#from cleverhans.attacks import CarliniWagnerL2

import csv
import scipy.io
import glob
import numpy as np
import sys
from EOT_tile import EOT_ATTACK
import math

# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

keras.layers.core.K.set_learning_phase(0)
# loading model
#set session config
#session_conf = tf.ConfigProto(
#      intra_op_parallelism_threads=2,
#     inter_op_parallelism_threads=2)
sess = tf.Session()
K.set_session(sess)

print("Loading model")    
model = load_model('./ResNet_30s_34lay_16conv.hdf5')
#model = load_model('weights-best_k0_r0.hdf5')

wrap = KerasModelWrapper(model)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
    x =  x[0, 0:maxlen]
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

def op_concate(x, w, i):
    data_len = 9000
    tile_times = math.ceil(data_len/w)
    x_tile = np.tile(x, (1, tile_times, 1))
    p = i
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]

    return np.append(x2, x1, axis=1)

preds = model(x)

## Loading time serie signals
id = int(sys.argv[1])
count = id-1
record = "A{:05d}".format(id)
local_filename = dataDir+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))

X_test=np.float32(data)
new_X_test = np.repeat(data, 5, axis=0)

target = np.zeros((1, 1))
target = np.array([int(sys.argv[2])])
target_a = utils.to_categorical(target, num_classes=4)

ground_truth_a = utils.to_categorical(ground_truth, num_classes=4)
dis_metric = int(sys.argv[3])

start_time = time.time()
perturb_window = 400#int(sys.argv[4])
ensemble_size = 30#int(sys.argv[5])

eotl2 = EOT_ATTACK(wrap, sess=sess)
eotl2_params = {'y_target': target_a, 'learning_rate': 1, 'max_iterations': 500, 'initial_const': 50000, 'perturb_window': perturb_window, 'dis_metric': dis_metric, 'ensemble_size': ensemble_size, 'ground_truth': ground_truth_a}
adv_x = eotl2.generate(x, **eotl2_params)
adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
feed_dict = {x: X_test}
adv_sample = adv_x.eval(feed_dict=feed_dict, session = sess)

print("time used:", time.time()-start_time)

perturb = adv_sample - X_test

perturb = perturb[:, 0:perturb_window, :]

'''
correct = 0
attack_success = np.zeros(4)
not_success_in_ensemble_size = 0
for i in range(perturb_window):
    if i == 0:
        test_all = zero_mean(op_concate(perturb, perturb_window, i)+X_test)
    else:
        test_all = np.append(test_all, zero_mean(op_concate(perturb, perturb_window, i)+X_test), axis=0)

prob_att = model.predict(test_all)
prob = model.predict(test_all)
ind = np.argmax(prob, axis=1)
for _, it in enumerate(ind):
    attack_success[it] = attack_success[it] + 1

print("attack success times:", attack_success)
'''

'''
import matplotlib.pyplot as plt
plt.figure()
plt.plot(perturb[0,:,0])
plt.show(block=False)

adv_sample = op_concate(perturb,perturb_window,False) + X_test
plt.figure()
plt.plot(adv_sample[0,1000:2000,0])
plt.show(block=False)

plt.figure()
plt.plot(X_test[0,1000:2000,0])
plt.show(block=False)
'''
perturb_squeeze = np.squeeze(perturb, axis=2)
if dis_metric == 1:
    outputstr = './output/EOTtile_w'+sys.argv[4]+'_e'+sys.argv[5]+'_l2_A'+sys.argv[1]+'T'+sys.argv[2]+'.out'
else:
    if dis_metric == 2:
        outputstr = './output/EOTtile_w'+sys.argv[4]+'_e'+sys.argv[5]+'_dtw_A' + sys.argv[1] + 'T' + sys.argv[2] + '.out'
    else:
        outputstr = './output/EOTtile_w' + sys.argv[4]+'_e'+sys.argv[5] + '_smooth_A' + sys.argv[1] + 'T' + sys.argv[2] + '.out'
np.savetxt(outputstr, perturb_squeeze,delimiter=",")
