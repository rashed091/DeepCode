import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


a = [1, 2, 3]
b = [1, 2, 3]
c = [4, 3, 1]

w = tf.equal(a, b)
x = tf.equal(b, c)

y = tf.reduce_all(w)
z = tf.reduce_any(x)
with tf.Session() as sess:
    print(sess.run(w))
    print(sess.run(y))
    print(sess.run(x))
    print(sess.run(z))


equality_casting = tf.cast(w, tf.float32)
reduce_mean = tf.reduce_mean(equality_casting)
reduce_sum = tf.reduce_sum(equality_casting)

with tf.Session() as sess:
    print(sess.run(equality_casting))
    print(sess.run(reduce_mean))
    print(sess.run(reduce_sum))
