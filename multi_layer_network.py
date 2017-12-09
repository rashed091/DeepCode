# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)

tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

# Neurons in each layers
l1 = 200
l2 = 100
l3 = 60
l4 = 30

# Weights
W1 = tf.Variable(tf.truncated_normal([784, l1], stddev=0.1))
b1 = tf.Variable(tf.zeros([l1]))
W2 = tf.Variable(tf.truncated_normal([l1, l2], stddev=0.1))
b2 = tf.Variable(tf.zeros([l2]))
W3 = tf.Variable(tf.truncated_normal([l2, l3], stddev=0.1))
b3 = tf.Variable(tf.zeros([l3]))
W4 = tf.Variable(tf.truncated_normal([l3, l4], stddev=0.1))
b4 = tf.Variable(tf.zeros([l4]))
W5 = tf.Variable(tf.truncated_normal([l4, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))


# Model
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + b1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + b4)
Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100


prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


learning_rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(100):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {X: batch_x, Y_: batch_y}

    sess.run(optimizer, feed_dict=train_data)

    a, _ = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print(str(i) + ": ********* epoch " + "--> train accuracy:" + str(a))

sess.close()




