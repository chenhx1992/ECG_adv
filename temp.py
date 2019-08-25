import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def bandPassFiltere(x):
    mask = tf.cast(tf.concat([tf.ones([1, 1]), tf.concat([tf.zeros([1, 3]), tf.zeros([1, 8189])], 1)], 1), dtype=tf.complex64)
    stfts = tf.contrib.signal.stft(tf.reshape(x,[9000]), data_len, 0,window_fn=None)
    stfts_masked = tf.multiply(tf.reshape(stfts,[1,8193]),mask)
    inverse_stfts = tf.contrib.signal.inverse_stft(stfts_masked, data_len,0, window_fn=None)
    #print(inverse_stfts.get_shape())
    #print(tf.reshape(inverse_stfts,[1,9000,1]).get_shape())
    return tf.reshape(inverse_stfts,[9000])
sess = tf.Session()

a = tf.range(1,901,1,dtype=tf.float32)
b = tf.range(1,1801,2,dtype=tf.float32)
c = tf.zeros([900])
b_sin = tf.sin(b)
c_sin = tf.sin(c)
b_stfts = tf.contrib.signal.stft(tf.reshape(b_sin,[900]), 900, 1,window_fn=None)
b_power = tf.reshape(tf.abs(b_stfts*tf.conj(b_stfts)),[513])
wave = c_sin#tf.add(tf.sin(a), b_sin)


stfts = tf.contrib.signal.stft(tf.reshape(wave,[900]), 900, 1,window_fn=None)
power = tf.reshape(tf.abs(stfts*tf.conj(stfts)),[513])

mask = tf.cast(tf.concat([tf.ones([1, 155]), tf.concat([tf.zeros([1, 16]), tf.ones([1, 342])], 1)], 1), dtype=tf.complex64)
stfts_masked = tf.multiply(tf.reshape(stfts,[1,513]),mask)
inverse_stfts = tf.contrib.signal.inverse_stft(stfts_masked, 900,1,window_fn=None)

new_stfts = tf.contrib.signal.stft(tf.reshape(inverse_stfts,[900]), 900, 1, window_fn=None)
new_power = tf.reshape(tf.abs(new_stfts*tf.conj(new_stfts)),[513])

np_wave = sess.run(new_power)
x_power = np.arange(513)

fig, ax = plt.subplots()

ax.plot(x_power,b_power.eval(session=sess))
ax.plot(x_power,np_wave)
plt.show()


x_signal = np.arange(900)


x_signal = np.arange(900)
plt.figure()
plt.plot(x_signal,b_sin.eval(session=sess))
plt.show()



plt.figure()
plt.plot(x_signal,inverse_stfts.eval(session=sess))
#ax.plot(x_power,np_wave)
plt.show()

######aaa