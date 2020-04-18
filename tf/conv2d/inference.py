# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784  
OUTPUT_NODE = 10  

IMAGE_SIZE = 28  
NUM_CHANNELS = 1  
NUM_LABELS = 10 


CONV1_DEEP = 32  
CONV1_SIZE = 5 


CONV2_DEEP = 64 
CONV2_SIZE = 5 


FC_SIZE = 512


def inference(X, regularizer, train=False):
    
    with tf.name_scope('layer_conv1'):
        conv1_weights = tf.Variable(
            tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], stddev=.1, dtype=tf.float32),
            name='weights')
        conv1_biases = tf.Variable(tf.constant(.01, shape=[CONV1_DEEP], dtype=tf.float32), name='biases')

        conv1 = tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1, ], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_biases)

    
    with tf.name_scope('layer_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    with tf.name_scope('layer_conv2'):
        conv2_weights = tf.Variable(
            tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], stddev=.1, dtype=tf.float32),
            name='weights')
        conv2_biases = tf.Variable(tf.constant(.01, shape=[CONV2_DEEP], dtype=tf.float32), name='biases')

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1, ], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_biases)

    
    with tf.name_scope('layer_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    pool_shape = pool2.get_shape().as_list()  
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])  
    with tf.name_scope('layer_fc1'):
        fc1_weights = tf.Variable(tf.truncated_normal([nodes, FC_SIZE], stddev=.1, dtype=tf.float32), 'weights')
        fc1_biases = tf.Variable(tf.constant(.1, shape=[FC_SIZE], dtype=tf.float32), 'biases')

        
        if regularizer is not None:
            tf.add_to_collection('regularization', regularizer(fc1_weights))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        
        if train:
            fc1 = tf.nn.dropout(fc1, .5)

    
    with tf.name_scope('layer_output'):
        fc2_weights = tf.Variable(tf.truncated_normal([FC_SIZE, NUM_LABELS], stddev=.1, dtype=tf.float32), 'weights')
        fc2_biases = tf.Variable(tf.constant(.1, shape=[NUM_LABELS], dtype=tf.float32), 'biases')

        if regularizer is not None:
            tf.add_to_collection('regularization', regularizer(fc2_weights))

        Ylogits = tf.matmul(fc1, fc2_weights) + fc2_biases

    return Ylogits
