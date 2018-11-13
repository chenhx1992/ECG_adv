


import tensorflow as tf


sess = tf.InteractiveSession()
tf_dtype = tf.as_dtype('int32')

# Some tensor we want to print the value of
rand_times = tf.expand_dims(tf.random_uniform((), 5, 10, dtype=tf.int32),axis=0)
a = tf.constant([[1,2,3]],dtype=tf_dtype)
b = tf.expand_dims(a,axis=2)
b_shape=tf.shape(b)

bb= tf.expand_dims(tf.shape(b)[1],axis=0)
c = tf.concat([tf.constant([0]),rand_times-bb], axis=0)
c = tf.expand_dims(c, axis=0)
d = tf.expand_dims(tf.constant([0,0]),axis=0)

c = tf.concat([d,c],axis=0)
c = tf.concat([c,d], axis=0)

m = tf.pad(b, c, "CONSTANT")
m_shape = tf.shape(m)
print(rand_times.eval())

print(c.eval())
print(m.eval())
print(m_shape.eval())
