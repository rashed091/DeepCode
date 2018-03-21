import tensorflow as tf

x = tf.placeholder(tf.float32, None)
y = tf.placeholder(tf.float32, None)

z = x + y

with tf.Session() as sess:
    value = sess.run(z, feed_dict={x: 1, y: 2})
    print(value)
