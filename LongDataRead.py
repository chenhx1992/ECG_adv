#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:26:17 2018

@author: chenhx1992
"""

import matplotlib.pyplot as plt
import numpy as np
import wfdb

# Read a wfdb record using the 'rdrecord' function into a wfdb.Record object.
#record = wfdb.rdrecord('./LongData/04015') 
#wfdb.plot_wfdb(record=record, title='Record 04015 from MIT-BIH') 
# Read part of a WFDB annotation file into a wfdb.Annotation object, and plot the samples
#annotation = wfdb.rdann('./LongData/04015', 'atr', sampto=9200000)
#annotation.fs = 250
#wfdb.plot_wfdb(annotation=annotation, time_units='minutes')

# Read a WFDB record and annotation. Plot all channels, and the annotation on top of channel 0.
#record1 = wfdb.rdrecord('./LongData/04015')
#annotation1 = wfdb.rdann('./LongData/04015', 'atr')
#wfdb.plot_wfdb(record=record1, annotation=annotation1,
#               title='Record 04015 from MIT-BIH',
#               time_units='seconds')

# Read certain channels and sections of the WFDB record using the simplified 'rdsamp' function
# Return a numpy array and a dictionary
# Example code:    
# signals, fields = wfdb.rdsamp('sample-data/s0010_re', channels=[14, 0, 5, 10], sampfrom=100, sampto=15000)
file_name='04048'
signals, fields = wfdb.rdsamp('./LongData/'+file_name, channels=[0, 1])
annotation = wfdb.rdann('./LongData/'+file_name, 'atr')
annotation_np = np.zeros((len(annotation.sample), 2))
annotation_np[:, 0] =  annotation.sample
fig, axs = plt.subplots(3, 1, figsize=(50,40), sharex=True)
axs[0].plot(signals[:,0])
axs[1].plot(signals[:,1])
axs[2].scatter(annotation_np[:,0], annotation_np[:,1], marker='+', facecolor='red')
