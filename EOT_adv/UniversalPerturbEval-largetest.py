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
    tile_times = math.ceil(data_len/w)
    x_tile = np.tile(x, (1, tile_times, 1))
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]
    return np.append(x2, x1, axis=1)


# parameters
dataDir = '../training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

print("Loading ground truth file")
csvfile = list(csv.reader(open('../REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

start_time = time.time()
print("Loading model")
model = load_model('../ResNet_30s_34lay_16conv.hdf5')


#loading perturbation
perturb_window = 200
ensemble_size = 30
ground_truth = int(sys.argv[1])
target = int(sys.argv[2])


if ground_truth == 0:
    target_file = np.genfromtxt('../data_select_A.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 1:
    target_file = np.genfromtxt('../data_select_N.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 2:
    target_file = np.genfromtxt('../data_select_O.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 3:
    target_file = np.genfromtxt('../data_select_i.csv', delimiter=',')
    target_id = target_file[:,3]
target_len = target_file[:,2]
perturbDir = '../output/'+str(ground_truth)+'/'
pattern = r'EOTtile_w200_e30_l2_A[0-9]+T'+str(target)+'.out'
pattern = r'EOTtile_w200_e30_l2_A422T'+str(target)+'.out'
attack_success_all = np.zeros((4),dtype=int)
for (_, _, filenames) in walk(perturbDir):
    for inputstr in filenames:
        if re.match(pattern,inputstr) != None:
            attack_success = np.zeros((4), dtype=int)
            print("input file: ", perturbDir+inputstr)
            perturb = genfromtxt(perturbDir+inputstr, delimiter=',')
            dist = np.linalg.norm(perturb)
            perturb = np.expand_dims(perturb, axis=0)
            perturb = np.expand_dims(perturb, axis=2)
            for i, id_float in enumerate(target_id):
                if int(target_len[i]) < 30:
                    continue
                   
                id_1 = int(id_float)
                count = id_1 - 1
                record_1 = "A{:05d}".format(id_1)

                # Loading record 1
                local_filename = dataDir + record_1
                mat_data = scipy.io.loadmat(local_filename)
                data = mat_data['val']
                data = preprocess(data, WINDOW_SIZE)
                X_test_1 = np.float32(data)

                # Generate test data
                for p in range(perturb_window):
                    if p == 0:
                        test_all = zero_mean(op_concate(perturb, perturb_window, p) + X_test_1)
                    else:
                        test_all = np.append(test_all, zero_mean(op_concate(perturb, perturb_window, p) + X_test_1), axis=0)

                #predict
                prob = model.predict(test_all)
                ind = np.argmax(prob, axis=1)
                attack_success_current = np.zeros((4),dtype=int)
                for _, it in enumerate(ind):
                    attack_success_current[it] = attack_success_current[it] + 1
                    attack_success[it] = attack_success[it] + 1
                    attack_success_all[it] = attack_success_all[it] + 1
                print("attack_success_current:",attack_success_current)
            print("attack success:", attack_success)

attack_success_all = attack_success_all/np.sum(attack_success_all)

print("attack success all:", attack_success_all)
print("time:",time.time()-start_time)


