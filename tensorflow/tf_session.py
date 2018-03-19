import tensorflow as tf

m1 = tf.constant([[2,2]])
m2 = tf.constant([[3],[3]])

dot_operation = tf.matmul(m1, m2)

with tf.Session() as sess:
    result = sess.run(dot_operation)
    print(result)