import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


values = [10, 0, 30, 40, 50, 60, 3, 1, 2]


mn = tf.argmin(values, 0)
mx = tf.argmax(values, 0)

with tf.Session() as sess:
    print(sess.run(mn))
    print(sess.run(mx))

    