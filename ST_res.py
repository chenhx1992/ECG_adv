#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:51:12 2018

@author: chenhx1992
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt

folder = './Subjective_offline/'
a0 = genfromtxt(folder + 'order_A_N_0_9.csv', delimiter=',')
a1 = genfromtxt(folder + 'order_A_N_10_19.csv', delimiter=',')
a2 = genfromtxt(folder + 'order_A_N_20_29.csv', delimiter=',')
a3 = genfromtxt(folder + 'order_A_N_30_39.csv', delimiter=',')
a4 = genfromtxt(folder + 'order_A_N_40_49.csv', delimiter=',')
a5 = genfromtxt(folder + 'order_A_N_50_59.csv', delimiter=',')

a = np.vstack((a0, a1, a2, a3, a4, a5))

res = pd.read_excel('./offline_res/hqw.xlsx')
res = res.fillna(0)

diff = 0
l2 = 0
l2diff = 0

for i in range(0, 50):
    file_name = res.iloc[i, 0]
    fid = file_name.rstrip('_all.png')
    fid = fid.lstrip('R')
    fid = int(fid)
    # print("fid:{}".format(fid))
    
    labels = res.iloc[i, 1:4]
    labels = np.array(labels.values, dtype=int)
    # print(labels)
    
    decision = np.argmax(labels==1)+1    
    # print(decision)
    
    row_idx = np.argmax(a[:,0] == fid)
    # print(a[row_idx, :])
    col_idx = np.argmax(a[row_idx,1:4] == decision) + 1
    
    if (col_idx == 1):
        diff += 1
    
    if (col_idx == 2):
        l2 += 1
        print("fid:{}, Decision:{}".format(fid, decision))
    
    if (col_idx == 3):
        l2diff += 1

print("diff:{}".format(diff))
print("l2:{}".format(l2))
print("l2diff:{}".format(l2diff))