#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:53:22 2018

@author: chenhx1992
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt


filelist = np.array(genfromtxt('./Subjective_AMT/batch_N_A.txt'), dtype=int)
df = pd.DataFrame(index = filelist)

data = pd.read_csv('./Subjective_AMT/Batch_3410045_N_A_results.csv')

# ApprovedData = data[data['AssignmentStatus']=='Approved']
ApprovedData = data

ApprovedWorker = ApprovedData['WorkerId']

UniqueWorker = ApprovedWorker.unique()

numofWorker = UniqueWorker.shape[0]

choice_arr = ['optionA', 'optionB', 'optionC']
choices = ['Input.image_A_url', 'Input.image_B_url', 'Input.image_C_url']
    
count = 1
numofworkerlessthan10 = 0
for element in UniqueWorker:
    info = ApprovedData[ApprovedData['WorkerId'] == element][['WorkTimeInSeconds','Input.image_A_url', 'Input.image_B_url', 'Input.image_C_url', 'Answer.choice']]
    
    if info.shape[0] < 10:
        numofworkerlessthan10+=1
        continue
    
    diff = 0
    l2 = 0
    l2diff = 0
    print("{}th Worker {} complete {} tasks.".format(count, element, info.shape[0]))
    count =  count + 1
    # print(info)
    for index, rows in info.iterrows():
        
        a = rows['Answer.choice']
        idx = choice_arr.index(a)
        
        pic_name = info.loc[index,choices[idx]]
        if pic_name.startswith('KAQ'):
            continue
        
        print(pic_name)
        pic_idx, res = pic_name.split('_')
        pic_idx = pic_idx.lstrip('R')
        res = res.rstrip('.png')
        df.loc[int(pic_idx), element] = res
        
        if res == 'l2':
            l2+=1
        if res == 'diff':
            diff+=1
        if res == 'l2diff':
            l2diff+=1
    print('Choices: {} l2, {} diff, {} l2diff'.format(l2, diff, l2diff))
    # if count >=1:
        # break


df = df.drop(['max_occur_item','max_occur','total_occur'], axis=1)          

statistic = np.zeros((50, 4))
statistic[:,0] = filelist
count = 0
for index, rows in df.iterrows():
    # max_occur_item = df.loc[index].value_counts().idxmax()
    # max_occur = df.loc[index].value_counts().max()
    # total_occur = sum(df.loc[index].value_counts())
    # df.loc[index, 'max_occur_item'] = max_occur_item
    # df.loc[index, 'max_occur'] = max_occur
    # df.loc[index, 'total_occur'] = total_occur
    tmp = df.loc[index].value_counts()
    
    for row in tmp.index:
        if row == 'l2':
            statistic[count, 1] = tmp['l2']
        if row == 'diff':
            statistic[count, 2] = tmp['diff']
        if row == 'l2diff':
            statistic[count, 3] = tmp['l2diff']    
    count = count+1


import matplotlib.pyplot as plt
ind = np.arange(50)
width = 0.5
p1 = plt.bar(ind, statistic[:,1], width)
p2 = plt.bar(ind, statistic[:,2], width, bottom=statistic[:,1])
p3 = plt.bar(ind, statistic[:,3], width, bottom=statistic[:,2]+statistic[:,1])
plt.legend((p1[0], p2[0], p3[0]), ('L2-norm', 'Smooth', 'Smooth+L2-norm'))
plt.ylabel('Number of Times Selected')
plt.xlabel('Index of Test Signals')   

win_arr = np.zeros(3)
for i in range(50):
    tmp = statistic[i, 1:4]
    total = sum(tmp)
    max_val = np.max(tmp)
    max_idx = np.argmax(tmp)
    if max_idx == 0:
       print(i, statistic[i])
    if max_val > total*0.5:
        win_arr[max_idx] += 1       
    else:
        if len(np.argwhere(tmp==max_val)) == 1:
            win_arr[max_idx] += 1
        # else:
            # print(i, statistic[i])
    
print(win_arr)