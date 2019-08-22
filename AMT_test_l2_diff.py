#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:58:54 2018

@author: chenhx1992
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = pd.read_csv('./Subjective_AMT/FinalStage/Batch_3413930_l2_diff_results.csv')

data1 = pd.read_csv('./Subjective_AMT/FinalStage/Batch_212036_batch_results_Part2-1.csv') 
data2 = pd.read_csv('./Subjective_AMT/FinalStage/Batch_212037_batch_results_Part2-2.csv') 
data3 = pd.read_csv('./Subjective_AMT/FinalStage/Batch_212038_batch_results_Part2-3.csv') 
data4 = pd.read_csv('./Subjective_AMT/FinalStage/Batch_212039_batch_results_Part2-4.csv')

data = data.append(data1, ignore_index=True)
data = data.append(data2, ignore_index=True)
data = data.append(data3, ignore_index=True)
data = data.append(data4, ignore_index=True)

ApprovedData = data
# ApprovedData = data[data['AssignmentStatus']=='Approved']

ApprovedWorker = ApprovedData['WorkerId']
UniqueWorker = ApprovedWorker.unique()
numofWorker = UniqueWorker.shape[0]

new_data = ApprovedData[['HITId', 'WorkerId','WorkTimeInSeconds',
                 'Input.image_A1_url', 'Input.image_B1_url', 'Answer.choice1',
                 'Input.image_A2_url', 'Input.image_B2_url', 'Answer.choice2', 
                 'Input.image_A3_url', 'Input.image_B3_url','Answer.choice3', 
                 'Input.image_A4_url', 'Input.image_B4_url','Answer.choice4', 
                 'Input.image_A5_url', 'Input.image_B5_url','Answer.choice5', 
                 'Input.image_A6_url', 'Input.image_B6_url','Answer.choice6', 
                 'Input.image_A7_url', 'Input.image_B7_url','Answer.choice7', 
                 'Input.image_A8_url', 'Input.image_B8_url','Answer.choice8', 
                 'Input.image_A9_url', 'Input.image_B9_url','Answer.choice9', 
                 'Input.image_A10_url', 'Input.image_B10_url','Answer.choice10']]

worker_grouped = new_data.groupby(['WorkerId']).size()
# worker_grouped.loc['A11K4HY613LGHX']

for index, rows in new_data.iterrows(): 
    new_data.loc[index, 'numofassign'] =  worker_grouped.loc[new_data.loc[index, 'WorkerId']]
    
Adopted_arr = pd.DataFrame(UniqueWorker, columns=['WorkerId'])
Adopted_arr['AdoptedTimes'] = np.zeros((numofWorker,1))
Adopted_arr = Adopted_arr.set_index('WorkerId')

response_arr = pd.DataFrame(ApprovedData['HITId'].unique(), columns=['HITId'])
response_arr['ResponseTimes'] = np.zeros((ApprovedData['HITId'].unique().shape[0],1))
response_arr = response_arr.set_index('HITId')


HIT_grouped = new_data.groupby(['HITId'])

df = pd.DataFrame()
# df.loc['a', 'b'] = 1

for name, group in HIT_grouped:
    # print(name, group.shape, type(group))
    group = group.sort_values(by=['numofassign'])
    count = 0
    for index, rows in group.iterrows(): 
        if Adopted_arr.loc[group.loc[index, 'WorkerId'], 'AdoptedTimes'] >= 8:
            continue
        else:
            Adopted_arr.loc[group.loc[index, 'WorkerId'], 'AdoptedTimes'] +=1
            response_arr.loc[group.loc[index, 'HITId'],'ResponseTimes'] += 1
            for i in range(1, 11):
                val = 'Answer.choice' + str(i)
                ans = rows[val]
                ans = ans.lstrip('option')
                workerID = rows['WorkerId']
                val2 =  'Input.image_' + ans + str(i) + '_url'
                pic_name = rows[val2]
                prefix, pic_idx, source, target, prefer = pic_name.split('_')
                prefer = prefer.rstrip('.png')
                tmp = prefix + '_' + pic_idx + '_' + source + '_' + target
                df.loc[tmp, workerID] = prefer
            count +=1
            if count>=5:
                break
            
#### preference_question 
preference_question = pd.DataFrame()
count = 0
for index, rows in df.iterrows():
    # max_occur_item = df.loc[index].value_counts().idxmax()
    # max_occur = df.loc[index].value_counts().max()
    # total_occur = sum(df.loc[index].value_counts())
    # df.loc[index, 'max_occur_item'] = max_occur_item
    # df.loc[index, 'max_occur'] = max_occur
    # df.loc[index, 'total_occur'] = total_occur
    
    # preference_question.loc[count, 0] = int(index.split('_')[0])
    tmp = df.loc[index].value_counts()
    
    for row in tmp.index:
        if row == 'l2':
            preference_question.loc[index, 'l2'] = tmp['l2']
        if row == 'diff':
            preference_question.loc[index, 'diff'] = tmp['diff']  
    count = count+1
preference_question = preference_question.fillna(0)

# find diff is best 
for index, rows in preference_question.iterrows():
    if rows['diff'] == 5:
        print(index)
    
    
#### Final questions without 5 response for offline
f1 = open("./Subjective_AMT/TODO_l2_diff_1.csv", "w")
f2 = open("./Subjective_AMT/TODO_l2_diff_2.csv", "w")
f3 = open("./Subjective_AMT/TODO_l2_diff_3.csv", "w")
f4 = open("./Subjective_AMT/TODO_l2_diff_4.csv", "w")
f5 = open("./Subjective_AMT/TODO_l2_diff_5.csv", "w")
file_arr = [f1, f2, f3, f4, f5]
col_count_arr = np.zeros(5)

for index, rows in preference_question.iterrows():
    if rows['l2'] + rows['diff'] < 5:
        strs = []
        strs.append(index+'_l2.png')
        strs.append(index+'_diff.png')
        strs = np.array(strs)
        order = np.random.permutation(2)

        new_rows = strs[order]
    
        todotimes = int(5-(rows['diff'] + rows['l2']))
        for i in range(todotimes):
            file_arr[i].write(index+'_original.png,')
            for j in range(2):
                file_arr[i].write(str(new_rows[j]))
                if (j < 1):
                    file_arr[i].write(',')
            if col_count_arr[i] == 9:
                file_arr[i].write('\n')
                col_count_arr[i] = 0
            else:
                file_arr[i].write(',')
                col_count_arr[i] +=1            

for i in range(5):
    file_arr[i].close()
    
#### Statistic  
numofpartial = 0
diff_lwin = 0
diff_swin = 0
l2_lwin = 0
l2_swin = 0
count = 0
for index, rows in preference_question.iterrows():
    if rows['l2'] + rows['diff'] < 5:
        numofpartial += 1
    else:
        if rows['diff'] >=4:
            diff_lwin += 1
        if rows['diff'] == 3:
            diff_swin += 1
        if rows['l2'] >= 4:
            l2_lwin += 1
        if rows['l2'] == 3:
            l2_swin += 1
    count +=1

print(diff_lwin, diff_swin, l2_lwin, l2_swin, numofpartial)
print(count, diff_lwin+diff_swin+l2_lwin+l2_swin+numofpartial)

#### Plot 
ind = np.arange(600)
width = 0.5
p1 = plt.bar(ind, preference_question['l2'], width)
p2 = plt.bar(ind, preference_question['diff'], width, bottom=preference_question['l2'])
plt.legend((p1[0], p2[0]), ('L2-norm', 'Smooth'), frameon=False)
plt.ylabel('Number of Times Selected')
plt.xlabel('Index of Test Signals')   


#### preference_question_groupbyclass 
preference_question_groupbyclass = pd.DataFrame()
count = 0
for index, rows in df.iterrows():
    # max_occur_item = df.loc[index].value_counts().idxmax()
    # max_occur = df.loc[index].value_counts().max()
    # total_occur = sum(df.loc[index].value_counts())
    # df.loc[index, 'max_occur_item'] = max_occur_item
    # df.loc[index, 'max_occur'] = max_occur
    # df.loc[index, 'total_occur'] = total_occur
    
    # preference_question.loc[count, 0] = int(index.split('_')[0])
    source_target = index.split('_')[2] + '-' + index.split('_')[3]
    preference_question_groupbyclass.loc[index, 'source-target'] = source_target
    
    tmp = df.loc[index].value_counts()
    
    for row in tmp.index:
        if row == 'l2':
            preference_question_groupbyclass.loc[index, 'l2'] = tmp['l2']
        if row == 'diff':
            preference_question_groupbyclass.loc[index, 'diff'] = tmp['diff']  
    count = count+1

preference_question_groupbyclass = preference_question_groupbyclass.fillna(0)

preference_question_a_n = preference_question_groupbyclass[preference_question_groupbyclass['source-target'] == 'A-N']

preference_question_a_n = preference_question_groupbyclass
#
length=600
ind = np.arange(length)+1
diff_lw = np.zeros(length+2) + 3
diff_sw = np.zeros(length+2) + 2
l2_lw = np.zeros(length+2) - 3
l2_sw = np.zeros(length+2) - 2
zeros = np.zeros(length+2) 
width = 0.6
fig, axs = plt.subplots(1, 1, figsize=(160,120))
# p2 = plt.bar(ind, preference_question_a_n['diff'], width, bottom=preference_question_a_n['l2'])
# p1 = plt.bar(ind, preference_question_a_n['diff'], width, color='forestgreen',  alpha=0.8, hatch="\\")
# p2 = plt.bar(ind, -preference_question_a_n['l2'], width, color='steelblue', alpha=0.8, hatch="//")
p1 = plt.bar(ind, preference_question_a_n['diff'], width, color='forestgreen',  alpha=0.8)
p2 = plt.bar(ind, -preference_question_a_n['l2'], width, color='steelblue', alpha=0.8)
plt.plot(diff_lw, linewidth=2, color='black', linestyle='--')
plt.plot(diff_sw, linewidth=2, color='black', linestyle='-.')
plt.plot(l2_lw, linewidth=2, color='black', linestyle='--')
plt.plot(l2_sw, linewidth=2, color='black', linestyle='-.')
plt.plot(zeros, linewidth=2, color='black', linestyle='-')
plt.legend((p1[0], p2[0]), ('Smoothness', 'L2-norm'), frameon=False, bbox_to_anchor=(0., 1.0 , 1., .05), ncol=2, loc='upper center', fontsize=12)
plt.ylabel('Number of Times Selected', fontsize=12)
plt.xlabel('Index of Test Signals', fontsize=12)   
plt.yticks(np.arange(-5,6), [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
plt.xlim([0, length+1])
plt.ylim([-5,5])
fig.tight_layout()

#### preference_worker
preference_worker = pd.DataFrame()
count = 0
for column in df:
    cols = df[column]
    tmp = cols.value_counts()
    for row in tmp.index:
        if row == 'l2':
            preference_worker.loc[column, 'l2'] = tmp['l2']
        if row == 'diff':
            preference_worker.loc[column, 'diff'] = tmp['diff']  
    count = count+1   
preference_worker = preference_worker.fillna(0)

for index, rows in preference_worker.iterrows():
    preference_worker.loc[index, 'diff_percent'] = rows['diff'] / (rows['diff'] + rows['l2']) * 100
    preference_worker.loc[index, 'l2_percent'] = rows['l2'] / (rows['diff'] + rows['l2']) * 100
    
length = numofWorker
ind = np.arange(length)+1
up_w = np.zeros(length+2) - 50
down_w = np.zeros(length+2) + 50
zeros = np.zeros(length+2) 
fig, axs = plt.subplots(1, 1, figsize=(160,120))
width = 0.5
p1 = plt.bar(ind, preference_worker['diff_percent'], width, color='forestgreen',  alpha=0.8, hatch="\\")
p2 = plt.bar(ind, -preference_worker['l2_percent'], width, color='steelblue', alpha=0.8, hatch="-")
plt.plot(up_w, linewidth=1.5, color='black', linestyle='--')
plt.plot(down_w, linewidth=1.5, color='black', linestyle='--')
plt.plot(zeros, linewidth=1.5, color='black', linestyle='-')
plt.legend((p1[0], p2[0]), ('Smoothness', 'L2-norm'), frameon=False, bbox_to_anchor=(0., 1.0 , 1., .05), ncol=2, loc='upper center', fontsize=12)
plt.ylabel('Selection Percentage(%)', fontsize=12)
plt.xlabel('Index of Participants', fontsize=12)   
plt.yticks([-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
           [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xticks(np.arange(1, numofWorker+1))
plt.xlim([0, length+1])
plt.ylim([-100,100])
fig.tight_layout()



diff_prefer = 0
l2_prefer = 0
tie = 0
count = 0
for index, rows in preference_worker.iterrows():
    if rows['diff'] > rows['l2']:
        diff_prefer += 1
    else:
        if rows['diff'] < rows['l2']: 
            l2_prefer += 1
        else:
            tie +=1
    count +=1

print(diff_prefer, l2_prefer, tie)


df.to_csv('./paper_fig/ST_diff_l2_df.csv')
preference_question_groupbyclass.to_csv('./paper_fig/ST_diff_l2_preference_question_groupbyclass.csv')
preference_worker.to_csv('./paper_fig/ST_diff_l2_preference_worker.csv')
