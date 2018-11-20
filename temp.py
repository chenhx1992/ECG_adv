


import tensorflow as tf

print(tf.__version__)
sess = tf.InteractiveSession()
tf_dtype = tf.as_dtype('int32')

# Some tensor we want to print the value of

a = tf.constant(3)
b = tf.constant([1,2,3])
c = tf.shape(b)[0]

d = tf.cond(tf.equal(a,c),lambda: tf.constant(0), lambda: a)



print(c.eval())
print(a.eval())
print(d.eval())
