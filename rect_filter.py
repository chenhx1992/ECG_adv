import numpy as np
from keras.models import load_model
from random import randrange
from os import walk
import re
import csv
import glob
import scipy.io
from numpy import genfromtxt
import tensorflow as tf
import sys
from scipy import signal

def rectFilter(x, w):
    new_x = tf.reshape(tf.pad(tf.reshape(x,[1,w]), tf.constant([[0,0],[0,9000-w]]), "CONSTANT"), [9000])
    mask = tf.cast(tf.concat([tf.concat([tf.concat([tf.ones([1, 1]), tf.zeros([1, 3])], 1), tf.concat([tf.ones([1, 2727]), tf.zeros([1, 1])], 1)], 1),
                              tf.concat([tf.concat([tf.ones([1, 545]), tf.zeros([1, 1])], 1), tf.ones([1, 4915])], 1)],1), dtype=tf.complex64)
    stfts = tf.contrib.signal.stft(new_x, 9000, 1,window_fn=None)
    stfts_masked = tf.multiply(tf.reshape(stfts,[1,8193]),mask)
    inverse_stfts = tf.contrib.signal.inverse_stft(stfts_masked, 9000,1,window_fn=None)
    return tf.reshape(inverse_stfts,[9000])

sess = tf.InteractiveSession()
for i in range(4):
    perturbDir = './output/'+str(i)+'/'
    for (_, _, filenames) in walk(perturbDir):
        for inputstr in filenames:
            perturb = genfromtxt(perturbDir + inputstr, delimiter=',')
            perturb_window = int(len(perturb))
            filtered_perturb = rectFilter(tf.convert_to_tensor(perturb, dtype=tf.float32), perturb_window)
            new_perturb = sess.run(filtered_perturb)
            perturb = new_perturb[0:perturb_window]
            outputstr = perturbDir + "LDMF_recfilter"+inputstr[4:]
            np.savetxt(outputstr, perturb, delimiter=",")