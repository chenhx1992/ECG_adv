#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:20:06 2018

@author: chenhx1992
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

idx_file = './Subjective_AMT/distarr_A_N_0_360.csv'
data_select = genfromtxt(idx_file, delimiter=',')

l2 = data_select[:,1]
smooth = data_select[:,2]
smooth_part = data_select[:,4]
l2_part = data_select[:,3]


values, base = np.histogram(l2, bins=40)
cumulative = np.cumsum(values)
# plot the cumulative function
plt.plot(base[:-1], cumulative, c='lightblue')


classes = ['A', 'N', 'O','~']

colors = pl.cm.tab20(np.linspace(0,1,12))


count = 0
for item1 in classes:
    for item2 in classes:
        if item1 != item2:
            idx_file = './Subjective_AMT/distarr_' + item1 + '_' + item2 + '_0_360.csv'
            data_select = genfromtxt(idx_file, delimiter=',')
            l2 = data_select[:,1]
            # values, base = np.histogram(l2, bins=np.linspace(0.0, 380.0, num=1000))
            smooth = data_select[:,2]*10000
            # values, base = np.histogram(smooth, bins=np.linspace(0.0, 100.0, num=1000))
            smooth_part = data_select[:,4]
            l2_part = data_select[:,3]
            # values, base = np.histogram(smooth_part*10000+l2_part*0.01, bins=np.linspace(0.0, 100.0, num=1000))
            
            petl90 = np.percentile(smooth_part*10000+l2_part*0.01, 90)
            print(petl90)
            
            cumulative = np.cumsum(values)
            cumulative = cumulative/max(cumulative)
            # plot the cumulative function
            plt.plot(base[:-1], cumulative, color = colors[count], label = item1 + '-' + item2, linewidth=2.0)
            count += 1
plt.legend(ncol=2, frameon=False, fontsize=12)     
plt.xlabel('Distance')
plt.ylabel('Cumulative distribution function(CDF)')
plt.show()
