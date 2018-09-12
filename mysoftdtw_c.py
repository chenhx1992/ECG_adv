#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:05:58 2018

@author: chenhx1992
"""

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

from softdtwc import softdtwc

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
    
    print("before\n")
    print(X.shape)
    print(delta.shape)
    
    X = X.reshape(X.shape[1], X.shape[2])
    delta = delta.reshape(delta.shape[1], delta.shape[2])
    
    print("after\n")
    print(X.shape)
    print(delta.shape)
    
    D = euclidean_distances(X, X+delta, squared = True)
    m = D.shape[0]
    n = D.shape[1]
    d = X.shape[1]
    
    # add an extra row and an extra column to D
    # needed to deal with edge cases in the recursion
    D = np.vstack((D, np.zeros(n)))
    D = np.hstack((D, np.zeros((m+1, 1))))
    D = np.array(D, dtype=np.float32)
    
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    R = np.zeros((m+2, n+2), dtype=np.float32)
    E = np.zeros((m+2, n+2), dtype=np.float32)
    G = np.zeros((m, d), dtype=np.float32)
    
    print('hi1')
    softdtwc(X, delta, D, R, E, G, gamma)
    print('hi2')
    
    G = G.reshape(1, m, d)
    
    print('dtw:{}'.format(R[m, n]))
    print(G)
    
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    
    print('Grad Hi')
    G = op.outputs[1]
    temp = tf.constant(0)
    
    return grad_1*G, grad_1*G, temp