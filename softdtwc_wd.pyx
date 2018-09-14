#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:27:35 2018

@author: chenhx1992
"""

# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

#from cython.parallel import prange
cimport cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
np.import_array()


from libc.float cimport FLT_MAX
from libc.math cimport exp, log, pow
from libc.string cimport memset

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef inline float softmin3(float a, float b, float c, float gamma):
    
    a = a/-gamma
    b = b/-gamma
    c = c/-gamma
    
    cdef float max_val = max(max(a, b), c)
    
    cdef float tmp = 0
    tmp += exp(a-max_val)
    tmp += exp(b-max_val)
    tmp += exp(c-max_val)

    return -gamma * (log(tmp) + max_val)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
def softdtwc_wd(np.ndarray[float, ndim=2] X,
             np.ndarray[float, ndim=2] delta,
             np.ndarray[float, ndim=2] R,
             np.ndarray[float, ndim=2] E,
             np.ndarray[float, ndim=2] G, 
             float gamma,
             int window):
    
#    cdef np.ndarray Y = np.add(X, delta)
    
    # added an extra row and an extra column on the python side
    cdef int m = X.shape[0]
    cdef int n = delta.shape[0]
    cdef int d = X.shape[1] 
    
    cdef int i, j
    
    tD = np.zeros((m+1, n+1), dtype=np.float32)
    cdef float [:, :] D = tD
    
    for i in range(m):
        for j in range(n):
            D[i,j] = pow((X[i, 0] - X[j, 0] - delta[j, 0]), 2)
            
#    cdef np.ndarray R = np.zeros([m+2, n+2], dtype=np.float32) 
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    memset(<void*>R.data, 0, (m+2)*(n+2) * sizeof(float))
    
    cdef int wd = window
    
#    cdef np.ndarray[float, ndim=2] mask = np.zeros([m+2, n+2], dtype=np.float32)
#    cdef np.ndarray[np.float32_t, ndim=2] bias = np.zeros([m+2, n+2], dtype=np.float32)
    
#    mask = np.zeros((m+2, n+2), dtype=np.int32)
#    bias = np.zeros((m+2, n+2), dtype=np.float32)
#    cdef int [:, :] mask_view = mask
#    cdef float [:, :] bias_view = bias
#    
#    for i in range(1, m+1):
#        for j in range(max(i-wd, 1), min(i+wd+1, n+1)):
#            mask_view[i, j] = 1
#    
#    mask_view[0, 0] = 1 
    
#    bias_view[:, :] = FLT_MAX
#    print('b00:{}'.format(bias_view[0, 0]))

#    R = np.multiply(R, mask) + np.multiply(bias, np.abs(mask-1))
#    for i in range(m+2):
#        for j in range(n+2):
#            R[i,j] = R[i, j] * mask_view[i,j] + bias_view[i,j] * abs(mask_view[i,j]-1)
     
#    print('r00:{}'.format(R[0, 0]))
#    print('r01:{}'.format(R[0, 1]))
#    print('r11:{}'.format(R[1, 1]))
#    print('r1600:{}'.format(R[1, 600]))
#    print('r1601:{}'.format(R[1, 601]))
#    print('r1602:{}'.format(R[1, 602]))
#    print('r1000:{}'.format(R[100, 0]))
#    print('r0100:{}'.format(R[0, 100]))
    
    for i in range(m+1):
        R[i, 0] = FLT_MAX
        R[i, max(i-wd-1, 0)] = FLT_MAX
        R[i, min(i+wd+1, n+1)] = FLT_MAX

    for j in range(n+1):
        R[0, j] = FLT_MAX
    
    R[0, 0] = 0
 
    #DP recursion
    for i in range(1, m+1):
        for j in range(max(i-wd, 1), min(i+wd+1, n+1)):
            # D is indexed starting from 0
            R[i, j] = D[i-1, j-1] + softmin3(R[i-1, j], R[i-1, j-1], R[i, j-1], gamma)
    
#    print('dtw:{}'.format(R[m, n]))
    # --------------------- Grad calculate ---------------------
    
    memset(<void*>E.data, 0, (m+2)*(n+2)*sizeof(float))
    
#    R = np.multiply(R, mask) + np.multiply(-bias, np.abs(mask-1))
#    for i in range(m+2):
#        for j in range(n+2):
#            R[i,j] = R[i, j] * mask_view[i,j] - bias_view[i,j] * abs(mask_view[i,j]-1) 
    
#    print('dtw:{}'.format(R[m, n]))
#    print('r00:{}'.format(R[0, 0]))
#    print('r01:{}'.format(R[0, 1]))
#    print('r11:{}'.format(R[1, 1]))
#    print('r1600:{}'.format(R[1, 600]))
#    print('r1601:{}'.format(R[1, 601]))
#    print('r1602:{}'.format(R[1, 602]))
    
    for i in range(1, m+1):
        D[i-1, n] = 0
        R[i, n+1] = -FLT_MAX
        R[i, max(i-wd-1, 0)] = -FLT_MAX
        R[i, min(i+wd+1, n+1)] = -FLT_MAX
    
    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -FLT_MAX
        
    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0
       
    cdef float a, b, c
    # DP recursion
    for j in reversed(range(1, n+1)):
        for i in reversed(range(max(j-wd, 1), min(j+wd+1, m+1))):
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1, j+1] * c

#    E = E[1:-1, 1:-1]     
#    cdef np.ndarray[float, ndim=2] tE = np.zeros([m, n], dtype=np.float32)
#    tE = np.zeros((m, n), dtype=np.float32)
#    cdef float [:, :] tE_view = tE
#    for i in range(m):
#        for j in range(n):
#            tE_view[i, j] = E[i+1, j+1]
    
#    for i in range(m):
#        for j in range(max(i-wd, 0), min(i+wd+1, n)):
#            for k in range(d):
#                G[i, k] += E[i+1, j+1] * 2 * (X[i, k] - X[j, k] - delta[j, k])
    for i in range(m):
        for j in range(max(i-wd, 0), min(i+wd+1, n)):
            G[i, 0] += E[i+1, j+1] * 2 * (X[i, 0] - X[j, 0] - delta[j, 0])