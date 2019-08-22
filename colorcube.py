#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:06:56 2018

@author: chenhx1992
"""

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from itertools import combinations
import itertools

lst = np.arange(0, 256, 5)
# for combo in combinations(lst, 3):  # 2 for pairs, 3 for triplets, etc
#     print(combo)

RGBlist = list(itertools.product(lst, lst, lst))

# RGBlist = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(100)]
paleta=list(zip(*RGBlist))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter([(x-127)/128. for x in paleta[0]],[(x-127)/128. for x in paleta[1]],[(x-127)/128. for x in paleta[2]], c=[(r[0] / 255., r[1] / 255., r[2] / 255.) for r in RGBlist])
ax.grid(False)
ax.set_title('RGB Color Cube')
ax.set_xlabel('Red',fontsize=12)
ax.set_ylabel('Green',fontsize=12)
ax.set_zlabel('Blue',fontsize=12, rotation=90)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# plt.savefig('blah.png')