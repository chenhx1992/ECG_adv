


import tensorflow as tf

print(tf.__version__)
sess = tf.InteractiveSession()
tf_dtype = tf.as_dtype('int32')

# Some tensor we want to print the value of
rand_times = tf.random_uniform((), 5, 10, dtype=tf.int32)
rand_times = tf.constant(10)
start = tf.constant(1)
a = tf.range(start, rand_times)
b = tf.random_shuffle(a)


print(rand_times.eval())
print(a.eval())
print(b.eval())

