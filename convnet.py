from __future__ import absolute_import, division, print_function

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


print("Tensorflow version " + tf.__version__)

tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=False, reshape=True, validation_size=0)

# In memory training data for this simple case.
# When data is too large to fit in memory, use Tensorflow queues.
def train_data_input_fn():
    return tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],
                                  batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)

# Eval data is an in-memory constant here.
def eval_data_input_fn():
    return tf.constant(mnist.test.images), tf.constant(mnist.test.labels)


def conv_model_loss(Ylogits, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,10), Ylogits)) * 100
    else:
        None


def conv_model_train_op(loss, mode=''):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=0.003, optimizer='Adam',
                                               learning_rate_decay_fn=lambda lr, step: 0.0001 + tf.train.exponential_decay(lr, step, -2000, math.e))


def conv_model_eval_metrics(classes, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        return {'accuracy': tf.metrics.accuracy(classes, Y_)}


def conv_model(features, labels, mode):
    X = features
    Y = labels
    XX = tf.reshape(X, [-1, 28, 28, 1])

    l1 = tf.layers.conv2d(XX, filters=6, kernel_size=6, padding='same', activation=tf.nn.relu)
    l2 = tf.layers.conv2d(l1, filters=12, kernel_size=5, padding='same', strides=2, activation=tf.nn.relu)
    l3 = tf.layers.conv2d(l2, filters=24, kernel_size=4, padding='same', strides=2, activation=tf.nn.relu)

    Y_ = tf.reshape(l3, [-1, 7*7*24])

    dense = tf.layers.dense(inputs=Y_, units=200, activation=tf.nn.relu)

    dropout = tf.layers.dropout(dense, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    Ylogits = tf.layers.dense(dropout, 10)

    predict = tf.nn.softmax(Ylogits, name='softmax_tensor')
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, Y, mode)

    return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predict, 'classes': classes}, loss=loss,
                                      train_op=train_op, eval_metric_ops=eval_metrics)


def main():
    estimator = tf.estimator.Estimator(model_fn=conv_model, model_dir='checkpoints')

    tensor_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)

    estimator.train(input_fn=train_data_input_fn, steps=100, hooks=[logging_hook])

    eval_results = estimator.evaluate(input_fn=eval_data_input_fn, steps=1)
    print(eval_results)


if __name__ == '__main__':
    main()
