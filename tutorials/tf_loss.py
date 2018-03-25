import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

X = tf.linspace(-1., 1., 500)
Y = tf.constant(0.)


l1_loss = tf.abs(Y - X)
l1 = sess.run(l1_loss)


l2_loss = tf.square(Y - X)
l2 = sess.run(l2_loss)

delta = tf.constant(0.25)
phuber_loss = tf.multiply(tf.square(delta), tf.sqrt(
    1. + tf.square((Y - X) / delta)) - 1)
huber_loss = sess.run(phuber_loss)

x_array = sess.run(X)
plt.plot(x_array, l2, 'b-', label='L2 Loss')
plt.plot(x_array, l1, 'r--', label='L1 Loss')
plt.plot(x_array, huber_loss, 'k-.', label='P-Huber Loss (0.25)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
