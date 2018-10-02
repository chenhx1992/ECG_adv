#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:00:58 2018

@author: chenhx1992
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
#from softdtwc import softdtwc
from softdtwc_wd import softdtwc_wd
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
    
    print('dtw:{}'.format(R[m, n]))
#    print(G)
    
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    
    print('Grad Hi')
    temp = tf.constant(0)
    
    return grad_1*op.outputs[1], grad_1*op.outputs[1], temp


def mysquare_new(x, y, z, name=None):

    with ops.name_scope(name, "MysquareNew", [x, y, z]) as name:
        sqr_x, grad = py_func(square_with_grad,
                              [x, y, z],
                              [tf.float32, tf.float32],
                              name=name,
                              grad=_MySquareGradNew)
        return sqr_x

def square_with_grad(x, y, z):
  # first output: x, second output: gradient of sqr_x with respect to x
  print('square_with_grad')
  return np.square(y), 2 * y


def _MySquareGradNew(op, grad_sqr, grad_grad):
    # grad_sqr - gradient of some global function with respect to the first output of the op
    # grad_grad - gradient of some global function with respect to the second output of the op
    # op.outputs[0] - tensor that equals to op.inputs[0] * op.inputs[0]
    # op.outputs[1] - tensor that equals to 20 * op.inputs[0]
    print('_MySquareGradNew')
    return grad_sqr * op.outputs[1], grad_sqr * op.outputs[1], tf.constant(0)

import matplotlib.pyplot as plt

# Test code 1
#sess = tf.Session()
#
#tmp = np.array([1., 2.], dtype=np.float32).reshape((1, 2, 1))
#X = tf.constant(tmp)
#tmp = np.array([1., 1.], dtype=np.float32).reshape((1, 2, 1))
#delta = tf.constant(tmp)
#gamma = tf.constant(1.)
#
#Z = mysoftdtw(X, delta, gamma)
#
#tf.global_variables_initializer().run(session=sess)
#
#print(Z.eval(session=sess))
#print(tf.gradients(Z, delta)[0].eval(session=sess))

# Test code 2
sess = tf.Session()
length = 9000
tmp = np.linspace(0.0, 1.0, num=length, dtype=np.float32).reshape((1, length, 1))
tmp = 6 * np.sin(2 * np.pi * 20 * tmp)
tmp1 = np.float32(tmp)
#plt.plot(tmp[0,:])
#print(tmp)
X = tf.constant(tmp1)
#tmp2 = np.linspace(0.1, 0.1, num=length, dtype=np.float32).reshape((1, length, 1))
tmp2 = np.random.normal(0, 0.1, length).reshape((1, length, 1))
tmp2 = np.float32(tmp2)
#plt.plot(tmp[0,:])
delta = tf.constant(tmp2)
##delta = Y-X
gamma = tf.constant(0.1, dtype=tf.float32)
gamma2 = tf.constant(1.0, dtype=tf.float32)
#
Z = mysoftdtw(X, delta, gamma)
Z2 = mysoftdtw(X, delta, gamma2)
M = mysquare_new(X, delta, gamma)
#
tf.global_variables_initializer().run(session=sess)

plt.plot(tmp2[0,:])
plt.plot(tmp1[0,:])
plt.plot(tmp1[0,:]+tmp2[0,:])

start_time = time.time()
#print(start_time)
Z.eval(session=sess)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
#print(start_time)
Z2.eval(session=sess)
print("--- %s seconds ---" % (time.time() - start_time))

#start_time = time.time()
#print(tf.gradients(Z, delta)[0].eval(session=sess))
#print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(np.sum(M.eval(session=sess)))
print("--- %s seconds ---" % (time.time() - start_time))

# Test code 3
#import time
#sess = tf.Session()
#length = 9000
#x = tf.placeholder(tf.float32, shape = (1,length,1))
#y = tf.placeholder(tf.float32, shape = (1,length,1))
#gamma = tf.constant(1.)
#
#z = mysoftdtw(x, y, gamma)
#m = mysquare_new(x, y, gamma)
#tf.global_variables_initializer().run(session=sess)
#
#g = tf.gradients(z, [x, y, gamma], stop_gradients=[x, y, gamma])
#
#tmp1 = np.linspace(1.0, 10.0, num=length, dtype=np.float32).reshape((1, length, 1))
#tmp2 = np.linspace(1.0, 1.0, num=length, dtype=np.float32).reshape((1, length, 1))
##feed_dict = {x: np.array([1., 2.]).reshape((1, 2, 1)), y: np.array([1., 1.]).reshape((1, 2, 1))}
#feed_dict = {x: tmp1, y: tmp2}
#start_time = time.time()
#g[1].eval(session=sess, feed_dict=feed_dict)
#print("--- %s seconds ---" % (time.time() - start_time))
#
#print(np.sum(m.eval(session=sess, feed_dict=feed_dict)))

# Other test code
#g = tf.gradients(Z, [X, delta, gamma], stop_gradients=[X, delta, gamma])
##print(tf.gradients(Z, delta, stop_gradients = [X, delta])[0].eval(session=sess))
#print(tf.gradients(Z, delta)[0].eval(session=sess))
    