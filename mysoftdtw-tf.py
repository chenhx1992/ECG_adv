#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:03:24 2018

@author: chenhx1992
"""


# ------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics.pairwise import euclidean_distances
from math import exp, log
import numpy as np

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
                             [np.float32, np.float32],
                             name = name,
                             grad = softdtwGrad)
        return sdtw

# TODO: test difference between python and cpython
def softmin3(a, b, c, gamma):
    print('softdtw3-3')
    a /= -gamma
    b /= -gamma
    c /= -gamma
    
    max_val = max(max(a, b), c)
    
    tmp = 0
    tmp += exp(a-max_val)
    tmp += exp(b-max_val)
    tmp += exp(c-max_val)
    print('softdtw3-4')
    return np.float32(-gamma * (log(tmp) + max_val))


def softdtw(X, delta, gamma):
    
#    print("before\n")
#    print(X.shape)
#    print(delta.shape)
    print('softdtw')
    X = X.reshape(X.shape[1], X.shape[2])
    delta = delta.reshape(delta.shape[1], delta.shape[2])
    print('softdtw0')
#    print("after\n")
#    print(X.shape)
#    print(delta.shape)
    
    D = euclidean_distances(X, X+delta, squared = True)
    
    m = D.shape[0]
    n = D.shape[1]
    print('softdtw1')
    # add an extra row and an extra column to D
    # needed to deal with edge cases in the recursion
    D = np.vstack((D, np.zeros(n)))
    D = np.hstack((D, np.zeros((m+1, 1))))
    D = np.array(D, dtype=np.float32)
    print('softdtw2')
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    R = np.zeros((m+2, n+2), dtype=np.float32)
    
    for i in range(m+1):
        R[i, 0] = np.finfo(np.float32).max

    for j in range(n+1):
        R[0, j] = np.finfo(np.float32).max
    
    R[0, 0] = 0
    print('softdtw3')
    #DP recursion
    for i in range(1, m+1):
        for j in range(1, n+1):
            # D is indexed starting from 0
            print('softdtw3-1')
            R[i, j] = D[i-1, j-1] + softmin3(R[i-1, j], R[i-1, j-1], R[i, j-1], gamma)
            print('softdtw3-2')
            
    print('softdtw4')
    # --------------------- Grad calculate ---------------------
    Y = X+delta
    
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the recursion  
    E = np.zeros((m+2, n+2), dtype=np.float32)
    print('softdtw5')
    for i in range(1, m+1):
        D[i-1, n] = 0
        R[i, n+1] = -np.finfo(np.float32).max
    
    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -np.finfo(np.float32).max
        
    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0
    print('softdtw6')
    # DP recursion
    for j in reversed(range(1, n+1)):
        for i in reversed(range(1, m+1)):
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1, j+1] * c
    
    E = E[1:-1, 1:-1]        
    d = X.shape[1]
    print('softdtw7')
    G = np.zeros((m, d), dtype=np.float32)
    
    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
    
    G = G.reshape(1, m, d)
#    G = tf.convert_to_tensor(G, np.float32)
    print('softdtw8')
#    print(X)
#    print(delta)
    print('dtw:{}'.format(R[m, n]))
#    print(G)
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    print('softdtwGrad')
    temp = tf.constant(0)
    
    return grad_1*op.outputs[1], grad_1*op.outputs[1], temp
# ------------------------------------------------------------------------------------