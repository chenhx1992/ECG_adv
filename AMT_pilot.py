#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:00:28 2018

@author: chenhx1992
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt


# filelist = np.array(genfromtxt('./Subjective_AMT/batch_N_A.txt'), dtype=int)
# df = pd.DataFrame(index = filelist)

data = pd.read_csv('./Subjective_AMT/PilotTest/Batch_3413310_pilot2_results.csv')
# data = pd.read_csv('./Subjective_AMT/PilotTest/Batch_3413268_piolot_results.csv')
# data = pd.read_csv('./Subjective_AMT/Stage2/Batch_3413404_stage2_results.csv')

new_data = data[['WorkerId','WorkTimeInSeconds','Answer.choice1','Answer.choice2', 'Answer.choice3', 'Answer.choice4', 'Answer.choice5', 'Answer.choice6', 'Answer.choice7', 'Answer.choice8', 'Answer.choice9', 'Answer.choice10']]

arr = np.zeros((80,1))

#BABBAABAAB
#AAAABABBAB

satisfy = 0

for index, rows in new_data.iterrows():
    diff = 0
    l2 = 0

    if rows['Answer.choice1'] == 'optionB':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice2'] == 'optionA':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice3'] == 'optionB':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice4'] == 'optionB':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice5'] == 'optionA':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice6'] == 'optionA':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice7'] == 'optionB':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice8'] == 'optionA':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice9'] == 'optionA':
        diff += 1
    else:
        l2 += 1
    if rows['Answer.choice10'] == 'optionB':
        diff += 1
    else:
        l2 += 1
    
    arr[index] = diff
    
    new_data.loc[index, 'diff'] = diff
    if (diff>=8):
        satisfy +=1
    print('WorkerId: {}, l2:{}, diff: {}'.format(rows['WorkerId'], l2, diff))

print(satisfy)


incorrect = np.zeros((10,1))
selected =new_data[new_data['diff']>=7]
for index, rows in selected.iterrows():
    if rows['Answer.choice1'] != 'optionB':
        incorrect[0] +=1
    if rows['Answer.choice2'] != 'optionA':
        incorrect[1] +=1
    if rows['Answer.choice3'] != 'optionB':
        incorrect[2] +=1
    if rows['Answer.choice4'] != 'optionB':
        incorrect[3] +=1
    if rows['Answer.choice5'] != 'optionA':
        incorrect[4] +=1
    if rows['Answer.choice6'] != 'optionA':
        incorrect[5] +=1
    if rows['Answer.choice7'] != 'optionB':
        incorrect[6] +=1
    if rows['Answer.choice8'] != 'optionA':
        incorrect[7] +=1
    if rows['Answer.choice9'] != 'optionA':
        incorrect[8] +=1
    if rows['Answer.choice10'] != 'optionB':
        incorrect[9] +=1
print(incorrect) 


df_worker = pd.read_csv('./Subjective_AMT/Stage2/User_1132931_workers.csv')
new_data = selected
for index, rows in new_data.iterrows():
    wid = new_data.loc[index, 'WorkerId']
    tmp = df_worker[df_worker['Worker ID']==wid].index
    # print(tmp)
    df_worker.loc[tmp, 'UPDATE-AllPass'] = 1


df_worker.to_csv('./Subjective_AMT/Stage2/User_1132931_workers1.csv')

