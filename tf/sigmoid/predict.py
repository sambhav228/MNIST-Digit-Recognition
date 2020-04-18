# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from .train import model_dir, model_name, mnist_dir


class Predict:
    def __init__(self, model_dir, model_name):
        with tf.gfile.GFile(os.path.join(model_dir, model_name), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='sigmoid')

        self.graph = graph

        self.X = graph.get_tensor_by_name('sigmoid/input/X:0')
        self.predict = graph.get_tensor_by_name('sigmoid/output/predict:0')

    def __call__(self, X):
        input = self.graph.get_tensor_by_name('sigmoid/input/X:0')
        predict = self.graph.get_tensor_by_name('sigmoid/output/predict:0')
        with tf.Session(graph=self.graph) as sess:
            pd = sess.run(predict, feed_dict={input: X})

        return pd


predict = Predict(model_dir, model_name)

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data as mnist_data

    mnist = mnist_data.read_data_sets(mnist_dir, one_hot=True)

    print(mnist.test.images[0])
    print(mnist.test.labels[0])
    print(predict([mnist.test.images[0]]))
