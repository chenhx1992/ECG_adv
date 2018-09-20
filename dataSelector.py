#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:10:22 2018

@author: chenhx1992
"""

# Select 1000 ECG samples for evalution

from numpy import genfromtxt
data_summary = genfromtxt('./DataAnalysis/prediction_correct.csv', delimiter=',')
import numpy as np

type_A = data_summary[(data_summary[:,0] == 0)]
type_A_sorted = type_A[type_A[:,2].argsort()][::-1]
#data_summary[(data_summary[:,0] == 0) & (data_summary[:,2] > 0.7)]
type_A_conf = type_A_sorted[0:250, :]
type_A_select = type_A_conf[:,[0,2,8]]

type_N = data_summary[(data_summary[:,0] == 1)]
type_N_sorted = type_N[type_N[:,2].argsort()][::-1]
type_N_conf = type_N_sorted[0:255, :]
type_N_select = type_N_conf[:,[0,2,8]]

type_O = data_summary[(data_summary[:,0] == 2)]
type_O_sorted = type_O[type_O[:,2].argsort()][::-1]
type_O_conf = type_O_sorted[0:250, :]
type_O_select = type_O_conf[:,[0,2,8]]

type_i = data_summary[(data_summary[:,0] == 3)]
type_i_sorted = type_i[type_i[:,2].argsort()][::-1]
type_i_conf = type_i_sorted[0:245, :]
type_i_select = type_i_conf[:,[0,2,8]]

data_select = np.vstack((type_A_select, type_N_select, type_O_select, type_i_select))

np.savetxt('data_select.csv', data_select, delimiter=",")