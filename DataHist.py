# -*- coding: utf-8 -*-
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import scipy.io
import numpy as np
import glob
import matplotlib.pyplot as plt

# Parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 60*FS

## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))
trainset = np.zeros((len(files),WINDOW_SIZE))
ECG_size_distribution = np.zeros((len(files),1))
count = 0
for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    print('Loading record {}'.format(record))    
    data = mat_data['val'].squeeze()
    # Preprocessing
#    print('Preprocessing record {}'.format(record))       
#    data = np.nan_to_num(data) # removing NaNs and Infs
#    data = data - np.mean(data)
#    data = data/np.std(data)
#    trainset[count,:min(WINDOW_SIZE,len(data))] = data[:min(WINDOW_SIZE,len(data))].T # padding sequence
    ECG_size_distribution[count, 0] = len(data)
    count += 1

plt.figure()
plt.hist(ECG_size_distribution/300, bins=300, cumulative=True, density=True, facecolor='g', alpha=0.75)
plt.xlabel('ECG sigal length(s)')
plt.ylabel('Culmulative Distribution')
plt.axis([9, 61, 0.0, 1.0])
plt.grid(True)
plt.show()

