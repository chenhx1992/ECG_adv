#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:42:30 2018

@author: chenhx1992
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.gridspec import GridSpec

ind = np.linspace(-np.pi, np.pi, 100)
X = np.sin(ind)
X = np.reshape(X, (1,100))
delta = np.random.randn(1, 100)*0.25


length = 500
f = 1.
w = 2. * np.pi * f
time_interval = 1
samples = length
t = np.linspace(0, time_interval, samples)
y = np.sin(w * t)
# plt.plot(t,y)
X = np.reshape(y, (1,length))

# f = 120.
# w = 2. * np.pi * f
# time_interval = 1
# samples = length
# t = np.linspace(0, time_interval, samples)
# y = 0.05*np.sin(w * t) + 0.05
# tmp = np.reshape(y, (1,length))

tri = (signal.triang(3)-0.5)*0.2
# tmp = np.tile(tri[0:2], 6)*0.5
delta = np.zeros((1,length))
random_arr = np.random.permutation(497)
# tri = np.array([0, 0.05, 0.1, 0.05, 0])
for i in range(0, 30):
    tmp = random_arr[i]
    if tmp < 250:
        delta[0,tmp:tmp+3] = -tri*1
    else:
        delta[0,tmp:tmp+3] = +tri*1

# delta[0,120:132] = tmp
# delta[0,245:257] = tmp
# delta[0,370:382] = tmp

# l2_norm = np.sum(np.square(delta))
# print(l2_norm)

# fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable

arr = [179,115,294,386,481,257,444,86,491,202,
       222,405,343,413,254,23,336,74,351,471,
       110,279,43,330,120,317,161,238,278,422]

fig = plt.figure()
# fig.suptitle("GridSpec w/ different subplotpars")
gs1 = GridSpec(12, 1)
gs1.update(left=0.1, right=0.9, wspace=0.05)
ax0 = plt.subplot(gs1[0:5, 0])
ax1 = plt.subplot(gs1[7:12, 0])
ax2 = plt.subplot(gs1[5, 0])
ax3 = plt.subplot(gs1[6, 0])
# ax4 = plt.subplot(gs1[12, 0])
# ax5 = plt.subplot(gs1[13, 0])
# ax6 = plt.subplot(gs1[14, 0])
# ax7 = plt.subplot(gs1[15, 0])
# ax8 = plt.subplot(gs1[16, 0])
# ax9 = plt.subplot(gs1[17, 0])
# ax10 = plt.subplot(gs1[18, 0])
# ax11 = plt.subplot(gs1[19, 0])

ax0.plot(X.T, color='black')
ax0.set_xlim([0,501])
# axs[4].plot(delta.T, color='black')
ax1.plot((X+delta).T, color='black')
ax1.set_xlim([0,500])
ax0.set_xlim([-5,504])
ax1.set_xlim([-5,504])
ax0.get_xaxis().set_visible(False)
# ax1.get_xaxis().set_visible(False)

cmp = 'gray'
a2 = ax2.imshow(X, cmap=cmp, vmin=-1, vmax=1)
ax2.set_aspect(10)
ax2.get_yaxis().set_visible(False)
a3 = ax3.imshow(X+delta, cmap=cmp, vmin=-1, vmax=1)
ax3.set_aspect(10)
ax3.get_yaxis().set_visible(False)
ax2.set_xlim([-5,504])
ax3.set_xlim([-5,504])
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)

cax = plt.axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(a3, cax=cax, ticks=[-1, 0, 1])

# fig.colorbar(a3, orientation='horizontal')
# cmp = 'autumn'
# ax4.imshow(X, cmap=cmp, vmin=-1, vmax=1)
# ax4.set_aspect(10)
# ax4.get_yaxis().set_visible(False)
# ax5.imshow(X+delta, cmap=cmp, vmin=-1, vmax=1)
# ax5.set_aspect(10)
# ax5.get_yaxis().set_visible(False)
# ax4.set_xlim([-5,504])
# ax5.set_xlim([-5,504])
# ax4.get_xaxis().set_visible(False)
# ax5.get_xaxis().set_visible(False)

# cmp = 'winter'
# ax6.imshow(X, cmap=cmp, vmin=-1, vmax=1)
# ax6.set_aspect(10)
# ax6.get_yaxis().set_visible(False)
# ax7.imshow(X+delta, cmap=cmp, vmin=-1, vmax=1)
# ax7.set_aspect(10)
# ax7.get_yaxis().set_visible(False)
# ax6.set_xlim([-5,504])
# ax7.set_xlim([-5,504])
# ax6.get_xaxis().set_visible(False)
# ax7.get_xaxis().set_visible(False)

# cmp = 'copper'
# ax8.imshow(X, cmap=cmp, vmin=-1, vmax=1)
# ax8.set_aspect(10)
# ax8.get_yaxis().set_visible(False)
# ax9.imshow(X+delta, cmap=cmp, vmin=-1, vmax=1)
# ax9.set_aspect(10)
# ax9.get_yaxis().set_visible(False)
# ax8.set_xlim([-5,504])
# ax9.set_xlim([-5,504])
# ax8.get_xaxis().set_visible(False)
# ax9.get_xaxis().set_visible(False)

# cmp = 'rainbow'
# ax10.imshow(X, cmap=cmp, vmin=-1, vmax=1)
# ax10.set_aspect(10)
# ax10.get_yaxis().set_visible(False)
# ax11.imshow(X+delta, cmap=cmp, vmin=-1, vmax=1)
# ax11.set_aspect(10)
# ax11.get_yaxis().set_visible(False)
# ax10.set_xlim([-5,504])
# ax11.set_xlim([-5,504])
# ax10.get_xaxis().set_visible(False)
# ax11.get_xaxis().set_visible(False)

plt.show()


