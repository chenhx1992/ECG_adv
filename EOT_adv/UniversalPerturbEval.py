import numpy as np
from keras.models import load_model
from random import randrange
import csv
import glob
import scipy.io
from numpy import genfromtxt
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

id_A = np.array([4,5,9,71,90,101,102,128,137,208,\
                 216,225,231,253,267,271,282,301,375,422,\
                 432,438,439,441,456,465,486,509,520,542])

id_N = np.array([1,2,3,6,7,10,12,14,16,18,\
                 19,21,25,26,28,32,33,35,36,37,\
                 40,44,45,46,48,49,50,51,52,53])

id_O = np.array([8,13,20,23,29,30,38,41,61,65,\
                 69,74,77,82,88,92,96,110,114,115,\
                 119,121,123,126,131,133,136,138,145,159])

id_i = np.array([22,34,56,106,125,139,164,201,205,307,\
                 370,445,474,524,537,585,591,619,629,690,\
                 700,705,761,774,813,984,988,1006,1048,1063])

#id_A = np.array([4])
#id_N = np.array([1])
print("Loading model")
model = load_model('../ResNet_30s_34lay_16conv.hdf5')
attack_success = np.zeros((30, 4),dtype=int)
attack_success_all = np.zeros((4),dtype=int)

#loading perturbation
perturb_window = 200
ensemble_size = 30
id_perturb = 5
target = 1

inputstr = '../output/EOTtile_w'+str(perturb_window)+'_e'+str(ensemble_size)+'_l2_A'+str(id_perturb)+'T'+str(target)+'.out'
print("input file: ", inputstr)
perturb = genfromtxt(inputstr, delimiter=',')
dist = np.linalg.norm(perturb)
print("distance:", dist)
perturb = np.expand_dims(perturb, axis=0)
perturb = np.expand_dims(perturb, axis=2)

for i, id_1 in enumerate(id_i):
    print(i)
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

    prob = model.predict(test_all)
    ind = np.argmax(prob, axis=1)
    for _, it in enumerate(ind):
        attack_success[i, it] = attack_success[i, it] + 1
        attack_success_all[it] = attack_success_all[it] + 1






#print("correct:", correct)
attack_success_all = attack_success_all/np.sum(attack_success_all)
print("attack success times:", attack_success_all)

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
