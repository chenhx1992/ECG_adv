import tensorflow as tf
sess = tf.InteractiveSession()

a = tf.constant([0,1,2,3,4])
rand_i = tf.expand_dims(tf.random_uniform((), 0, 5, dtype=tf.int32), axis=0)
p = tf.concat([rand_i, 5-rand_i], axis=0)
x1, x2 = tf.split(a, p, axis=0)
res = tf.concat([x2, x1], axis=0)
res = tf.reshape(res,[6])
print(res.shape)
print(sess.run(res))