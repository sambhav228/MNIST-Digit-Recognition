# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784  
OUTPUT_NODE = 10  


def get_weight_variable(shape, regularizer=None):
    
    weights = tf.get_variable("weight", initializer=tf.truncated_normal(shape, stddev=.1))

    
    if regularizer is not None:
        tf.add_to_collection('regularization', regularizer(weights))

    return weights


def inference(X, regularizer):
    
    with tf.name_scope('layer_out'):
        W = get_weight_variable([INPUT_NODE, OUTPUT_NODE], regularizer)
        b = tf.get_variable('biases', initializer=tf.constant(.1, shape=[OUTPUT_NODE]))

        Ylogits = tf.matmul(X, W) + b

        return Ylogits
