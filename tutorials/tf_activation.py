import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

relu = tf.nn.relu(x)
sigmoid = tf.nn.sigmoid(x)
tanh = tf.nn.tanh(x)
softplus = tf.nn.softplus(x)
softmax = tf.nn.softmax(x)

with tf.Session() as sess:
    y_relu, y_sigmoid, y_tanh, y_softmax, y_softplus = sess.run(
        [relu, sigmoid, tanh, softmax, softplus])

plt.figure('Plot', figsize=(8, 6))
plt.subplot(231)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(232)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(233)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(234)
plt.plot(x, y_softmax, c='red', label='softmax')
plt.ylim((-0.1, 1))
plt.legend(loc='best')

plt.subplot(235)
plt.plot(x, y_softmax, c='red', label='sofplus')
plt.ylim((-0.1, 1))
plt.legend(loc='best')

plt.show()
