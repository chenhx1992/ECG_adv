#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:16:02 2018

@author: chenhx1992
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pattern1 = re.compile("Loading record*")

df = pd.DataFrame(columns=['Truth', '1', '2', '3'])
count = 0
inner_count = 0
time_sum = 0
repeated = 0
folder = "./logfile/diff/"
for file in os.listdir(folder):
    print(file)
    f = open(folder+file, 'r')
    for line in f:
        if pattern1.match(line):
            idx = int(line.split(' ')[2].lstrip('A'))
            # print(idx)
        if 'Ground truth' in line:
            truth = int(line.split(':')[1])
            df.loc[idx, 'Truth'] = truth
            # print(truth) 
        if 'seconds' in line:
            tmp = float(line.split(' ')[1])
            time_sum += tmp
            count += 1
            if ~np.isnan(df.loc[idx, str(inner_count+1)]):
                print(idx)
                repeated +=1
            df.loc[idx, str(inner_count+1)] = tmp 
            inner_count = (inner_count + 1) % 3
            
tmp = df[['1', '2', '3']].sum(axis=1) /3 
mean = tmp.mean()
var = tmp.std()


# diff: 295.72487757224303
# l2: 96.5062147059239
# l2_diff: 177.7411653758318

avgs = np.zeros(3)
stds = np.zeros(3)

avgs[0] = 96.066
stds[0] = 12.987
avgs[1] = 177.431
stds[1] = 29.187
avgs[2] = 295.725
stds[2] = 35.375

ind = np.arange(3)  
width = 0.5

fig, axs = plt.subplots(1, 1, figsize=(40,40))
axs.bar(ind, avgs, width, yerr=stds, color='steelblue', alpha=0.5, error_kw=dict(ecolor='red', lw=1.5, capsize=5, capthick=2))
plt.xticks(ind, ['L2-norm', 'Smoothness+L2', 'Smoothness'], fontsize=12)
plt.ylim([0, 350])
axs.set_ylabel('Execution Time (seconds)', fontsize=12)
