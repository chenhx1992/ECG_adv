#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:50:52 2018

@author: chenhx1992
"""

import tensorflow as tf
from tensorflow.python.framework import ops
#from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
#from softdtwc import softdtwc
#from softdtwc_wd import softdtwc_wd
#import softdtwc

# Define custom py_func which takes also a grad op as argument 
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    #Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E8))
    
    tf.RegisterGradient(rnd_name)(grad)
    
    g = tf.get_default_graph()
    
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
    
# Define custom function using python
# in deep learning memory is always a bottleneck,thus using float32
def mysoftdtw(X, delta, gamma, name=None):
    with ops.name_scope(name, "mysoftdtw", [X, delta, gamma]) as name:
        sdtw, arr_G = py_func(softdtw,
                             [X, delta, gamma],
                             [tf.float32, tf.float32],
                             name = name,
                             grad = softdtwGrad)
        return sdtw


def softdtw(X, delta, gamma):
    
#    print("before\n")
#    print(X.shape)
#    print(delta.shape)
    
    X = X.reshape(X.shape[1], X.shape[2])
    delta = delta.reshape(delta.shape[1], delta.shape[2])
    
#    print("after\n")
#    print(X.shape)
#    print(delta.shape)
#    print("--- %s seconds ---" % time.time())
#    D = euclidean_distances(X, X+delta, squared = True)
#    print("--- %s seconds ---" % time.time())
    m = X.shape[0]
    n = delta.shape[0]
    d = X.shape[1]
    
    # add an extra row and an extra column to D
    # needed to deal with edge cases in the recursion
#    D = np.vstack((D, np.zeros(n)))
#    D = np.hstack((D, np.zeros((m+1, 1))))
#    D = np.array(D, dtype=np.float32)
#    print("--- %s seconds ---" % time.time())
    
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    R = np.zeros((m+2, n+2), dtype=np.float32)
    E = np.zeros((m+2, n+2), dtype=np.float32)
    G = np.zeros((m, d), dtype=np.float32)
    
#    print("--- %s seconds ---" % time.time())
    
    softdtwc_wd(X, delta, R, E, G, gamma, 300)
    
#    print("--- %s seconds ---" % time.time())  
    G = G.reshape(1, m, d)
    
#    print('dtw:{}'.format(R[m, n]))
#    print(G)
    
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    
    print('Grad Hi')
    temp = tf.constant(0)
    
    return grad_1*op.outputs[1], grad_1*op.outputs[1], temp