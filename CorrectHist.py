#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:58:12 2018

@author: chenhx1992
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('./DataAnalysis/prediction_correct.csv', sep=',',header=None)
array = df.values

type_a = array[array[:,0] == 0]
type_n = array[array[:,0] == 1]
type_o = array[array[:,0] == 2]
type_s = array[array[:,0] == 3]


type_distribution = [type_a.shape[0], type_n.shape[0], type_o.shape[0], type_s.shape[0]]
axix_label = ['A', 'N', '0', '~']

plt.bar(axix_label, type_distribution, align='center', alpha= 0.5)
plt.xlabel('Type name')
plt.ylabel('Sample size')
plt.ylim([0, 4500])
plt.grid()
plt.title('Type Distribution of correct prediction')
for i in range(4):
    plt.text(x = i-0.2 , y = type_distribution[i] + 100, s = str(type_distribution[i]), size = 15)



plt.figure()
plt.hist(type_o[:,7], bins=500, cumulative=True, density=True, facecolor='g', alpha=0.5)
plt.xlabel('ECG sigal length(s)')
plt.ylabel('Culmulative Distribution')
plt.title('Type O')
plt.axis([9, 61, 0.0, 1.0])
plt.grid(True)
plt.show()


