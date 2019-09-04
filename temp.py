import numpy as np
from random import randrange
from os import walk
import re
import csv
import glob
import scipy.io
from numpy import genfromtxt
import sys
from scipy import signal
import matplotlib.pyplot as plt
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



def filter(x):
    fs = 300

    #butterworth
    b, a = signal.butter(3, 0.05, btype='hp')
    bandpss_x = signal.lfilter(b, a, x)

    #notch filter
    f0 = 60
    b, a = signal.iirnotch(f0, 30, fs)
    y = signal.lfilter(b, a, bandpss_x)
    f0 = 50
    b, a = signal.iirnotch(f0, 30, fs)
    y = signal.lfilter(b, a, y)
    return bandpss_x

def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

def op_concate(x,w,p):
    if w != 9000:
        x_tile = np.tile(filter(x), (1, 1, 1))
        new_x = np.zeros((1,9000,1))
        new_x[0,p:p+w,0] = x_tile[0,:,0]
    else:
        x_tile = np.tile(x, (1, 1, 1))
        x1 = x_tile[:, 0:p, :]
        x2 = x_tile[:, p:9000, :]
        new_x = np.append(x2, x1, axis=1)
    return new_x

# parameters
dataDir = '../training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

print("Loading ground truth file")
csvfile = list(csv.reader(open('./REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))




#loading perturbation
perturb_window = 9000
ensemble_size = 30
ground_truth = 3
target = 1
id = 7382
if perturb_window == 9000:
    maxpos = 9000
else:
    maxpos = 9000-perturb_window

if ground_truth == 0:
    target_file = np.genfromtxt('./data_select_A.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 1:
    target_file = np.genfromtxt('./data_select_N.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 2:
    target_file = np.genfromtxt('./data_select_O.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 3:
    target_file = np.genfromtxt('./data_select_i.csv', delimiter=',')
    target_id = target_file[:,3]
target_len = target_file[:,2]
perturbDir = './output/LDMF/'+str(ground_truth)+'/'
pattern = r'LDMF_w'+str(perturb_window)+'_e30_l2_A'+str(id)+'T'+str(target)+'.out'
attack_success_all = np.zeros((4),dtype=int)

for (_, _, filenames) in walk(perturbDir):
    for inputstr in filenames:
        if re.match(pattern,inputstr) != None:
            attack_success = np.zeros((4), dtype=int)
            print("input file: ", perturbDir+inputstr)
            perturb = genfromtxt(perturbDir+inputstr, delimiter=',')
            dist = np.linalg.norm(perturb)
            plt.figure()
            plt.plot(perturb)
            plt.show()
            perturb = filter(perturb)
            plt.figure()
            plt.plot(perturb)
            plt.show()





