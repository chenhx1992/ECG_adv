import numpy as np
from keras.models import load_model
from random import randrange
from os import walk
import re
import csv
import glob
import scipy.io
from numpy import genfromtxt
import math
import sys
import time


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


def op_concate(x, w, p):
    data_len = 9000
    tile_times = math.ceil(data_len / w)
    x_tile = np.tile(x, (1, tile_times, 1))
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]
    return np.append(x2, x1, axis=1)


# parameters
dataDir = '../training_raw/'
FS = 300
WINDOW_SIZE = 30 * FS  # padding window for CNN
classes = ['A', 'N', 'O', '~']

print("Loading ground truth file")
csvfile = list(csv.reader(open('../REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir + "*.mat"))

# loading perturbation
perturb_window = 200
ensemble_size = 30
ground_truth = 2#int(sys.argv[1])
target = 1#int(sys.argv[2])

if ground_truth == 0:
    target_file = np.genfromtxt('../data_select_A.csv', delimiter=',')
    target_id = target_file[:, 3]
if ground_truth == 1:
    target_file = np.genfromtxt('../data_select_N.csv', delimiter=',')
    target_id = target_file[:, 3]
if ground_truth == 2:
    target_file = np.genfromtxt('../data_select_O.csv', delimiter=',')
    target_id = target_file[:, 3]
if ground_truth == 3:
    target_file = np.genfromtxt('../data_select_i.csv', delimiter=',')
    target_id = target_file[:, 3]
target_len = target_file[:, 2]
perturbDir = '../output/' + '2To1' + '/'
pattern = r'EOTtile_w200_e30_l2_A[0-9]+T' + str(target) + '.out'
attack_success_all = np.zeros((4), dtype=int)
all_dist = np.zeros(10)
k = 0
for (_, _, filenames) in walk(perturbDir):
    for inputstr in filenames:
        if re.match(pattern, inputstr) != None:
            attack_success = np.zeros((4), dtype=int)
            print("input file: ", perturbDir + inputstr)
            perturb = genfromtxt(perturbDir + inputstr, delimiter=',')
            dist = np.linalg.norm(perturb)
            dist = dist * dist
            all_dist[k] = dist/perturb_window*9000
            k = k + 1
            perturb_window = len(perturb)
            perturb = np.expand_dims(perturb, axis=0)
            perturb = np.expand_dims(perturb, axis=2)
            print(perturb_window)
print(all_dist)
print("mean distance", np.mean(all_dist))


