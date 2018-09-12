#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 18:57:19 2018

@author: chenhx1992
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from math import exp, log


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

# TODO: test difference between python and cpython
def softmin3(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma
    
    max_val = max(max(a, b), c)
    
    tmp = 0
    tmp += exp(a-max_val)
    tmp += exp(b-max_val)
    tmp += exp(c-max_val)

    return np.float32(-gamma * (log(tmp) + max_val))


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
    
    # add an extra row and an extra column to D
    # needed to deal with edge cases in the recursion
    D = np.vstack((D, np.zeros(n)))
    D = np.hstack((D, np.zeros((m+1, 1))))
    D = np.array(D, dtype=np.float32)
    
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    R = np.zeros((m+2, n+2), dtype=np.float32)
    
    for i in range(m+1):
        R[i, 0] = np.finfo(np.float32).max

    for j in range(n+1):
        R[0, j] = np.finfo(np.float32).max
    
    R[0, 0] = 0
    
    #DP recursion
    for i in range(1, m+1):
        for j in range(1, n+1):
            # D is indexed starting from 0
            R[i, j] = D[i-1, j-1] + softmin3(R[i-1, j], R[i-1, j-1], R[i, j-1], gamma)
    
    # --------------------- Grad calculate ---------------------
    Y = X+delta
    
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the recursion  
    E = np.zeros((m+2, n+2), dtype=np.float32)
    
    for i in range(1, m+1):
        D[i-1, n] = 0
        R[i, n+1] = -np.finfo(np.float32).max
    
    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -np.finfo(np.float32).max
        
    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0
    
    # DP recursion
    for j in reversed(range(1, n+1)):
        for i in reversed(range(1, m+1)):
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1, j+1] * c
    
    E = E[1:-1, 1:-1]        
    d = X.shape[1]
    
    G = np.zeros((m, d), dtype=np.float32)
    
    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
    
    
    G = G.reshape(1, m, d)
#    G = tf.convert_to_tensor(G, np.float32)
    
    print(X)
    print(delta)
    print('dtw:{}'.format(R[m, n]))
    print(G)
    
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    
#    X = op.inputs[0]
#    delta = op.inputs[1]
#    
#    print("before-grad\n")
#    print(X.shape)
#    print(delta.shape)
#    
#    X = tf.reshape(X, [X.shape[1], X.shape[2]])
#    delta = tf.reshape(delta, [delta.shape[1], delta.shape[2]])
#    
#    print("after-grad\n")
#    print(X.shape)
#    print(delta.shape)
#    
#    Y = tf.add_n([X, delta])
#    
#    R = op.outputs[1]
#    D = op.outputs[2]
#    gamma = op.inputs[2]
#    
#    m = X.shape[0]
#    n = delta.shape[0]
#    
#    print(m)
#    print(n)
#    
#    # need +2 because we use indices starting from 1
#    # and to deal with edge cases in the recursion
#    
#    E = np.zeros((m+2, n+2), dtype=np.float32)
#    
#    for i in range(1, m+1):
#        D[i-1, n] = 0
#        R[i, n+1] = -np.finfo(np.float32).max
#    
#    for j in range(1, n+1):
#        D[m, j-1] = 0
#        R[m+1, j] = -np.finfo(np.float32).max
#        
#    E[m+1, n+1] = 1
#    R[m+1, n+1] = R[m, n]
#    D[m, n] = 0
#    
#    # DP recursion
#    for j in reversed(range(1, n+1)):
#        for i in reversed(range(1, m+1)):
#            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
#            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
#            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
#            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1, j+1] * c
#    
#    E = E[1:-1, 1:-1]        
#    d = X.shape[1]
#    
#    G = np.zeros((m, d), dtype=np.float32)
#    
#    for i in range(m):
#        for j in range(n):
#            for k in range(d):
#                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
#    
#    G = G.reshape(1, m, d)
#    
#    print(G)
#    
#    G = tf.convert_to_tensor(G, np.float32)
    
    print('Grad Hi')
    G = op.outputs[1]
    temp = tf.constant(0)
    
    return grad_1*G, grad_1*G, temp

# Test code
#sess = tf.Session()
#
#tmp = np.array([1., 2.]).reshape((1, 2, 1))
#X = tf.constant(tmp)
#tmp = np.array([1., 1.]).reshape((1, 2, 1))
#delta = tf.constant(tmp)
###delta = Y-X
##gamma = tf.constant(1.)
##
##Z = mysoftdtw(X, delta, gamma)
##
##tf.global_variables_initializer().run(session=sess)
##
###print(Z.eval(session=sess))
##g = tf.gradients(Z, [X, delta, gamma], stop_gradients=[X, delta, gamma])
###print(tf.gradients(Z, delta, stop_gradients = [X, delta])[0].eval(session=sess))
##print(tf.gradients(Z, delta)[0].eval(session=sess))
#
#x = tf.placeholder(tf.float32, shape = (1,2,1))
#y = tf.placeholder(tf.float32, shape = (1,2,1))
#gamma = tf.constant(1.)
#
#z = mysoftdtw(x, y, gamma)
#g = tf.gradients(z, [x, y, gamma], stop_gradients=[x, y, gamma])
#feed_dict = {x: np.array([1., 2.]).reshape((1, 2, 1)), y: np.array([1., 1.]).reshape((1, 2, 1))}
#g[1].eval(session=sess, feed_dict=feed_dict)

# time test
sess = tf.Session()

len = 1000
tmp = np.linspace(1.0, 10.0, num=len, dtype=np.float32).reshape((1, len, 1))
#print(tmp)
X = tf.constant(tmp)
tmp = np.linspace(1.0, 2.0, num=len, dtype=np.float32).reshape((1, len, 1))
#print(tmp)
delta = tf.constant(tmp)
##delta = Y-X
gamma = tf.constant(1., dtype=tf.float32)
#
Z = mysoftdtw(X, delta, gamma)
#
tf.global_variables_initializer().run(session=sess)

import time
start_time = time.time()
print(Z.eval(session=sess))
print("--- %s seconds ---" % (time.time() - start_time))
    
    