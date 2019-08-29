import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import utils

import csv
import scipy.io
import glob
import numpy as np
import sys
from LDMF_EOT import LDMF_EOT_ATTACK
import math
import random

# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30 * FS  # padding window for CNN
classes = ['A', 'N', 'O', '~']

keras.layers.core.K.set_learning_phase(0)

sess = tf.Session()
K.set_session(sess)

print("Loading model")
model = load_model('./ResNet_30s_34lay_16conv.hdf5')

wrap = KerasModelWrapper(model)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

# load groundTruth
print("Loading ground truth file")
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir + "*.mat"))


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


def op_concate(x, w, i):
    data_len = 9000
    tile_times = math.ceil(data_len / w)
    x_tile = np.tile(x, (1, tile_times, 1))
    p = i
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]

    return np.append(x2, x1, axis=1)


ground_truth = int(sys.argv[1])
select_data_A = [6713,2509,6911,3852,869]
select_data_N = [3299,3344,3478,7125,4487]
select_data_O = [2170,5269,1241,7659,428]
#select_data_i = [4966,4020,7382,34,4941]
select_data_i = [4941]
#select_data_A = [6713,2509,6911]
#select_data_N = [3297,3344,3478]
#select_data_O = [2164,5276,1245]
#select_data_i = [4966,4020,7356]


#select_data_A = [7092, 4431, 5140, 6736, 621]
#select_data_N = [6946, 3748, 3353, 717, 1533]
#select_data_O = [6630, 1518, 6629, 1690, 1983]
#select_data_i = [34, 4948, 7794, 3738, 2693]
if ground_truth == 0:
    select_data = select_data_A
if ground_truth == 1:
    select_data = select_data_N
if ground_truth == 2:
    select_data = select_data_O
if ground_truth == 3:
    select_data = select_data_i


perturb_windows_set = [9000,7500,1500]

for _, target_id in enumerate(select_data):
    for _, perturb_window in enumerate(perturb_windows_set):
        if perturb_window == 9000:
            if target_id == 4966 or target_id == 4020 or target_id == 3344 or target_id == 2509 or target_id == 6713:
                continue
        id = int(target_id)

        count = id - 1
        record = "A{:05d}".format(id)
        local_filename = dataDir + record

        # Loading
        mat_data = scipy.io.loadmat(local_filename)
        print('Loading record {}'.format(record))
        #    data = mat_data['val'].squeeze()
        data = mat_data['val']
        data = preprocess(data, WINDOW_SIZE)
        X_test = np.float32(data)
        for i in range(4):
            if i == ground_truth:
                continue

            target = np.zeros((1, 1))
            target[0, 0] = int(i)
            target_a = utils.to_categorical(target, num_classes=4)
            ground_truth_a = utils.to_categorical(ground_truth, num_classes=4)

            dis_metric = 1
            if perturb_window != 9000:
                ensemble_size = int((9000 - perturb_window) / 50) #int(max(30, (9000 - perturb_window) / 50))
            else:
                ensemble_size = int(9000 / 50)

            eotl2 = LDMF_EOT_ATTACK(wrap, sess=sess)
            eotl2_params = {'y_target': target_a, 'learning_rate': 1, 'max_iterations': 500, 'initial_const': 50000,
                            'perturb_window': perturb_window, 'dis_metric': dis_metric, 'ensemble_size': ensemble_size,
                            'ground_truth': ground_truth_a}
            adv_x = eotl2.generate(x, **eotl2_params)
            adv_x = tf.stop_gradient(adv_x)  # Consider the attack to be constant
            feed_dict = {x: X_test}
            adv_sample = adv_x.eval(feed_dict=feed_dict, session=sess)

            perturb = adv_sample - X_test

            perturb = perturb[:, 0:perturb_window, :]

            perturb_squeeze = np.squeeze(perturb, axis=2)
            if dis_metric == 1:
                outputstr = './output/' + str(ground_truth) + '/LDMF_w' + str(perturb_window) + '_e30_l2_A' + str(
                    int(id)) + 'T' + str(int(target[0, 0])) + '.out'
            else:
                outputstr = './output/' + str(ground_truth) + '/EOTtile_w200_e30_smooth_A' + str(int(id)) + 'T' + str(
                    int(target[0, 0])) + '.out'
            np.savetxt(outputstr, perturb_squeeze, delimiter=",")


