#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:07:39 2018

@author: chenhx1992
"""

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]

def tf_diff_axis_1(a):
    return a[:,1:]-a[:,:-1]

import numpy as np
import tensorflow as tf

x0=np.expand_dims(np.arange(9),axis=0)
x0=np.expand_dims(x0,axis=2)
x0=np.float64(x0)
sess = tf.Session()
#np.diff(x0, axis=0) == sess.run(tf_diff_axis_0(tf.constant(x0)))
#np.diff(x0, axis=1) == sess.run(tf_diff_axis_1(tf.constant(x0)))

X = tf.constant(x0)
Z1 = tf.concat([X, [[[0]]]], 1)
Z6,Z7 = tf.split(Z1, tf.constant([5,5],dtype = tf.int32), axis=1)
Z8 = tf.concat([Z7, Z6], 1)
tf.gradients(Z8, Z1)[0].eval(session=sess)
 
Z2 = tf.concat([[[[0]]], X], 1) 
Z3 = Z1-Z2
Z4 = tf.slice(Z3, [0, 1, 0], [1, 8, 1])
tmp = np.zeros((1, 9001, 1), dtype=np.float32)
tmp[:,1:9000,:]=1
tmp = tf.constant(tmp)
Z5 = tf.multiply(Z3, tmp)
mean, var = tf.nn.moments(Z5, axes=[1])
var.eval(session=sess)
tf.gradients(var, X)[0].eval(session=sess)

Z8.eval(session=sess)
