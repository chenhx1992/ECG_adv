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

from softdtwc import softdtwc
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
    
    softdtwc(X, delta, D, R, E, G, gamma)
        
    G = G.reshape(1, m, d)
    
    print('dtw:{}'.format(R[m, n]))
#    print(G)
    
    return R[m, n], G


def softdtwGrad(op, grad_1, grad_2):
    
    print('Grad Hi')
    G = op.outputs[1]
    temp = tf.constant(0)
    
    return grad_1*G, grad_1*G, temp


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


# Test code 1
sess = tf.Session()

tmp = np.array([1., 2.], dtype=np.float32).reshape((1, 2, 1))
X = tf.constant(tmp)
tmp = np.array([0., 0.], dtype=np.float32).reshape((1, 2, 1))
delta = tf.constant(tmp)
gamma = tf.constant(1.)

Z = mysoftdtw(X, delta, gamma)

tf.global_variables_initializer().run(session=sess)

print(Z.eval(session=sess))
print(tf.gradients(Z, delta)[0].eval(session=sess))

# Test code 2
len = 1000
tmp = np.linspace(1.0, 10.0, num=len, dtype=np.float32).reshape((1, len, 1))
#print(tmp)
X = tf.constant(tmp)
tmp = np.linspace(0.0, 0.0, num=len, dtype=np.float32).reshape((1, len, 1))
#print(tmp)
delta = tf.constant(tmp)
##delta = Y-X
gamma = tf.constant(1., dtype=tf.float32)
#
Z = mysoftdtw(X, delta, gamma)
M = mysquare_new(X, delta, gamma)
#
tf.global_variables_initializer().run(session=sess)

import time
start_time = time.time()
print(Z.eval(session=sess))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(np.sum(M.eval(session=sess)))
print("--- %s seconds ---" % (time.time() - start_time))

# Other test code
#g = tf.gradients(Z, [X, delta, gamma], stop_gradients=[X, delta, gamma])
##print(tf.gradients(Z, delta, stop_gradients = [X, delta])[0].eval(session=sess))
#print(tf.gradients(Z, delta)[0].eval(session=sess))

#x = tf.placeholder(tf.float32, shape = (1,2,1))
#y = tf.placeholder(tf.float32, shape = (1,2,1))
#gamma = tf.constant(1.)
#
#z = mysoftdtw(x, y, gamma)
#g = tf.gradients(z, [x, y, gamma], stop_gradients=[x, y, gamma])
#feed_dict = {x: np.array([1., 2.]).reshape((1, 2, 1)), y: np.array([1., 1.]).reshape((1, 2, 1))}
#g[1].eval(session=sess, feed_dict=feed_dict)


    
    