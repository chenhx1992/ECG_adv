#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:29:45 2018

@author: chenhx1992
"""

from pathlib import Path
import numpy as np
from numpy import genfromtxt
import pandas as pd

# img_list = './Subjective_AMT/PilotTest/pilotlist.csv'
img_list = './Subjective_AMT/list.csv'
# TRUTH = '1'
# TARGET = '0'
# data_select = genfromtxt(idx_file, delimiter=',')
list_arr = pd.read_csv(img_list, header=None)
# classes = ['A', 'N', 'O','~']
# g_label = classes[int(TRUTH)]
# t_label = classes[int(TARGET)]

start = 0
end = 600
length = int((end-start)/10.0)
row_count = 0
col_count = 0
order_arr = np.zeros((length, 30), dtype=int)

# f = open("./Subjective_AMT/PilotTest/index_pilot_l2diff_diff.csv", "w")
f = open("./Subjective_AMT/index_diff_l2diff.csv", "w")

for i in range(start,end):
    prefix = list_arr.loc[i].values[0]
    
    order = np.random.permutation(2)
    
    order_arr[int(row_count/10.0), 3*col_count] = prefix.split('_')[1]
    order_arr[int(row_count/10.0), (3*col_count+1):(3*col_count+3)] = order
    
    rows = []
    rows.append(prefix+'_diff.png')
    rows.append(prefix+'_l2diff.png')
    # rows.append('R' + str(idx) + '_l2diff.png')
    
    rows = np.array(rows)

    new_rows = rows[order]
    
    f.write(prefix+'_original.png,')
    for j in range(2):
        f.write(str(new_rows[j]))
        if (j < 1):
            f.write(',')
    if col_count == 9:
        f.write('\n')
        col_count = 0
    else:
        f.write(',')
        col_count+=1
    
    
    
    row_count += 1    
    '''
    idx = int(data_select[i, 3]) 
    prefix = image_path + 'PilotTest/' + 'R' + str(idx) 
    my_file = Path(prefix + '_diff.png')
    
    if my_file.is_file():
        print("fid:{}, count:{}".format(idx, count))
        
        order = np.random.permutation(2)
        
        rows = []
        rows.append('R' + str(idx) + '_diff.png')
        rows.append('R' + str(idx) + '_l2.png')
        # rows.append('R' + str(idx) + '_l2diff.png')
        
        rows = np.array(rows)
    
        new_rows = rows[order]
        
        f.write('R' + str(idx) + '_original.png,')
        for j in range(2):
            f.write(str(new_rows[j]))
            if (j < 1):
                f.write(',')
        f.write('\n')
        
        order_arr[count, 0] = idx
        order_arr[count, 1:3] = order
        count += 1
    else:
        continue
    '''
    
# np.savetxt('./Subjective_AMT/PilotTest/order_pilot_l2diff_diff.csv', order_arr, delimiter=',')     
np.savetxt('./Subjective_AMT/order_diff_l2diff.csv', order_arr, delimiter=',')     
f.close()      
    