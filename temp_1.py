import matplotlib.pyplot as plt
import numpy as np
import scipy

c = np.arange(900)
d = c * 2
c_sin = np.sin(c)
d_sin = np.sin(d)
w = c_sin + d_sin
w_fft = scipy.fft(w,1025)

mask = np.concatenate((np.ones(155),np.zeros(16),np.ones(342),np.ones(342),np.zeros(16),np.ones(154)), 0)
masked_fft = mask*w_fft
masked_ifft = scipy.ifft(masked_fft)

new_fft = scipy.fft(masked_ifft,1025)
x = np.arange(513)
plt.figure()
plt.plot(x,np.abs(new_fft[0:513]))
#ax.plot(x_power,np_wave)
plt.show()
