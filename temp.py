import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
tf_dtype = tf.as_dtype('float32')

# Some tensor we want to print the value of
a = tf.constant([[11],[12],[11],[12],[13],[11],[12]],dtype=tf_dtype)

b = tf.nn.softmax(a,axis=0)

c = tf.constant([11,12,11,12,13,11,12],dtype=tf_dtype)
d = tf.nn.softmax(c)
print(b.eval())
print(d.eval())
