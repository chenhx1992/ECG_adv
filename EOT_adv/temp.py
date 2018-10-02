import tensorflow as tf
sess = tf.InteractiveSession()



a = tf.constant([[[1],[2],[3],[4],[5]],[[1],[2],[3],[4],[5]]])
data_mean, data_var = tf.nn.moments(a, axes=1)
mean = tf.expand_dims(tf.tile(data_mean,[1,5]),2)
var = tf.expand_dims(tf.tile(data_var,[1,5]),2)
b = (a-mean)/var
print(b.eval())