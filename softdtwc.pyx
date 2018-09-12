# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:20:04 2018

@author: chenhx1992
"""
# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


cimport cython
import numpy as np
cimport numpy as np
np.import_array()


from libc.float cimport FLT_MAX
from libc.math cimport exp, log
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
def softdtwc(np.ndarray[float, ndim=2] X,
             np.ndarray[float, ndim=2] delta,
             np.ndarray[float, ndim=2] D, 
             np.ndarray[float, ndim=2] R,
             np.ndarray[float, ndim=2] E,
             np.ndarray[float, ndim=2] G, 
             float gamma):
    
#    cdef np.ndarray Y = np.add(X, delta)
    
    # added an extra row and an extra column on the python side
    cdef int m = D.shape[0]-1
    cdef int n = D.shape[1]-1
    cdef int d = X.shape[1] 
    
    cdef int i, j
    
#    cdef np.ndarray R = np.zeros([m+2, n+2], dtype=np.float32) 
    # need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion
    memset(<void*>R.data, 0, (m+2)*(n+2) * sizeof(float))
    
    for i in range(m+1):
        R[i, 0] = FLT_MAX

    for j in range(n+1):
        R[0, j] = FLT_MAX
    
    R[0, 0] = 0
    
    cdef float z
    #DP recursion
    for i in range(1, m+1):
        for j in range(1, n+1):
            # D is indexed starting from 0
            R[i, j] = D[i-1, j-1] + softmin3(R[i-1, j], R[i-1, j-1], R[i, j-1], gamma)
    
    print('dtw:{}'.format(R[m, n]))
    # --------------------- Grad calculate ---------------------
    
    memset(<void*>E.data, 0, (m+2)*(n+2)*sizeof(float))
    
    for i in range(1, m+1):
        D[i-1, n] = 0
        R[i, n+1] = -FLT_MAX
    
    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -FLT_MAX
        
    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0
    
    cdef float a, b, c
    cdef float tmp
    # DP recursion
    for j in reversed(range(1, n+1)):
        for i in reversed(range(1, m+1)):
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1, j+1] * c
    
    cdef np.ndarray[float, ndim=2] tE = np.zeros([m+1, n+1], dtype=np.float32)
    
    for i in range(m+1):
        for j in range(n+1):
            tE[i, j] = E[i+1, j+1]
#    E = E[1:-1, 1:-1]        
    
    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += tE[i, j] * 2 * (X[i, k] - X[j, k] - delta[j, k])