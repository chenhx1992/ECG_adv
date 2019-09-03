import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()
a = tf.constant([1,2,3])
b = tf.constant([[0,0],[0,5]])
c = tf.reshape(tf.pad(tf.reshape(a,[1,3]),b,"CONSTANT"),[8])

print(sess.run(c))