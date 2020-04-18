# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784  
OUTPUT_NODE = 10  

L = 200
M = 100
N = 60
O = 30


def get_weight_variable(shape, regularizer=None):
    
    weights = tf.Variable(tf.truncated_normal(shape, stddev=.1))

    
    if regularizer is not None:
        tf.add_to_collection('regularization', regularizer(weights))

    return weights


def inference(X, regularizer):
    
    with tf.name_scope('layer'):
        W1 = get_weight_variable([INPUT_NODE, L], regularizer)
        b1 = tf.Variable(tf.constant(.1, shape=[L]))
        W2 = get_weight_variable([L, M], regularizer)
        b2 = tf.Variable(tf.constant(.1, shape=[M]))
        W3 = get_weight_variable([M, N], regularizer)
        b3 = tf.Variable(tf.constant(.1, shape=[N]))
        W4 = get_weight_variable([N, O], regularizer)
        b4 = tf.Variable(tf.constant(.1, shape=[O]))
        W5 = get_weight_variable([O, OUTPUT_NODE], regularizer)
        b5 = tf.Variable(tf.constant(.1, shape=[OUTPUT_NODE]))

        Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
        Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
        Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + b3)
        Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + b4)
        Ylogits = tf.matmul(Y4, W5) + b5

        return Ylogits
