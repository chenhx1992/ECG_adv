import tensorflow as tf
sess = tf.InteractiveSession()

a = tf.expand_dims(tf.constant([[0,1,2,3,4],[0,1,2,3,4]]),axis=2)
b = tf.expand_dims(tf.constant([[1,1,1,1,1]]),axis=2)
print(a.shape)
print(sess.run(a+b))