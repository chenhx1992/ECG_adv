import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

# Some tensor we want to print the value of
a = tf.constant([[11],[12],[11],[12],[13],[11],[12]])

b = tf.nn.softmax(a)

c = tf.constant([11,12,11,12,13,11,12])
d = tf.nn.softmax(c)
print(b.eval())
print(d.eval())
