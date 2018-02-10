import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

values = [1, 2, 3, 4]

x = tf.reshape(values, [-1, 2])

with tf.Session() as sess:
    print(sess.run(x))


s = tf.stack([tf.square(values), tf.ones_like(values)], 1)

with tf.Session() as sess:
    print(sess.run(s))
