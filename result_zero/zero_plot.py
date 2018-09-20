#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:22:01 2018

@author: chenhx1992
"""

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

o_A_l2 = genfromtxt('zero_tarA_l2.out', delimiter=',')
o_A = genfromtxt('zero_tarA.out', delimiter=',')
o_N_l2 = genfromtxt('zero_tarN_l2.out', delimiter=',')
o_N = genfromtxt('zero_tarN.out', delimiter=',')
o_O_l2 = genfromtxt('zero_tarO_l2.out', delimiter=',')
o_O = genfromtxt('zero_tarO.out', delimiter=',')

original = np.zeros((9000, 1), dtype=np.float64)

fig, axs = plt.subplots(4, 1, figsize=(50,40), sharex=True)
#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35)
plt.subplots_adjust(hspace=0.3)
axs[0].plot(original, color='black', label='Original data')
axs[0].set_title('Classification result: ~')
axs[0].set_ylim([-1, 1])
axs[0].set_xlim([-200, 9200])
#axs[0].set_xlabel('index')
axs[0].set_ylabel('signal value')
axs[0].legend(loc='upper center', frameon=False)

axs[1].plot(o_A, color='lightblue', label='Adversarial sample(soft-dtw)')
axs[1].plot(o_A_l2, color='salmon', label='Adversarial sample(L2-norm)')
axs[1].set_title('Classification result: A')
axs[1].set_ylim([-1, 1])
axs[1].set_xlim([-200, 9200])
#axs[0].set_xlabel('index')
axs[1].set_ylabel('signal value')
axs[1].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1), ncol=2)

axs[2].plot(o_N, color='lightblue', label='Adversarial sample(soft-dtw)')
axs[2].plot(o_N_l2, color='salmon', label='Adversarial sample(L2-norm)')
axs[2].set_title('Classification result: N')
axs[2].set_ylim([-1, 1])
axs[2].set_xlim([-200, 9200])
#axs[1].set_xlabel('index')
axs[2].set_ylabel('signal value')
axs[2].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1), ncol=2)

axs[3].plot(o_O, color='lightblue', label='Adversarial sample(soft-dtw)')
axs[3].plot(o_O_l2, color='salmon', label='Adversarial sample(L2-norm)')
axs[3].set_title('Classification result: O')
axs[3].set_ylim([-1, 1])
axs[3].set_xlim([-200, 9200])
axs[3].set_xlabel('sample index')
axs[3].set_ylabel('signal value')
axs[3].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1), ncol=2)

