import tensorflow as tf

x = tf.Variable(0)

add_opr = tf.add(x, 1)
update_opr = tf.assign(x, add_opr)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_opr)
        print(sess.run(x))
