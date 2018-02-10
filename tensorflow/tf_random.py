import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

z = tf.add(tf.random_normal([5,1]), tf.random_normal([5, 1]))

with tf.Session() as sess:
    print(sess.run(z))

    