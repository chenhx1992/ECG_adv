import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# loading perturbation
filename = './output/LDMF_w9000_l2_A6713T2.out'
perturb = genfromtxt(filename, delimiter=',')

plt.plot(perturb)
plt.show()


