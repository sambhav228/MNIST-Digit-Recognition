# -*- coding: utf-8 -*-
import os
import shutil

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import graph_util

from .inference import INPUT_NODE, OUTPUT_NODE, inference

# Learning Rate
LEARNING_RATE_BASE = .8
LEARNING_RATE_DECAY = .99  


TRAINING_STEPS = 30000
BATCH_SIZE = 100


REGULARIZATION_RATE = .0001


mnist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_name = 'model.pb'


def train(mnist):
    
    with tf.name_scope('input'):
        
        X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
        XX = tf.reshape(X, [-1, 784])
        
        Y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='Y_')

    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    with tf.name_scope('output'):
        
        Ylogits = inference(XX, regularizer)
        Y = tf.nn.softmax(Ylogits, name='predict')

    with tf.name_scope('loss'):
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
        
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('regularization'))

    
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('train'):
        
        decayed_learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           global_step,
                                                           mnist.train.num_examples / BATCH_SIZE,
                                                           LEARNING_RATE_DECAY,
                                                           staircase=True)

        
        train_step = tf.train.GradientDescentOptimizer(decayed_learning_rate) \
            .minimize(loss, global_step=global_step)

        
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.arg_max(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        validate_feed = {X: mnist.validation.images, Y_: mnist.validation.labels}
        test_feed = {X: mnist.test.images, Y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={X: xs, Y_: ys})

            if i % 1000 == 0:
                print(
                    f'step *** {sess.run(global_step):<7} '
                    f'validate accuracy *** {sess.run(accuracy, feed_dict=validate_feed):<0.20f}'
                )

        graph_def = sess.graph.as_graph_def()
        predict_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['output/predict'])
        with tf.gfile.GFile(os.path.join(model_dir, model_name), 'wb') as f:
            f.write(predict_graph_def.SerializeToString())
        print(predict_graph_def.node)

        # saver.save(sess, os.path.join(model_dir, model_name))

        print(
            f'step *** {sess.run(global_step):<7} '
            f'test accuracy *** {sess.run(accuracy, feed_dict=test_feed):<0.20f}'
        )


def main(argv=None):
    print('=== relu_5_layers_train ===')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    mnist = mnist_data.read_data_sets(mnist_dir, one_hot=True, reshape=False)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
