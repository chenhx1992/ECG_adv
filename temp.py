import tensorflow as tf

def bandPassFiltere(x, mask):
    stfts = tf.contrib.signal.stft(tf.reshape(x,[9000]), data_len, 0)

    stfts_masked = tf.multiply(tf.reshape(stfts,[1,8193]),mask)
    inverse_stfts = tf.contrib.signal.inverse_stft(stfts_masked, data_len,0)
    #print(inverse_stfts.get_shape())
    #print(tf.reshape(inverse_stfts,[1,9000,1]).get_shape())
    return tf.reshape(inverse_stfts,[9000])



sess = tf.Session()