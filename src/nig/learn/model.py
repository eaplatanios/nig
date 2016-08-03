import abc
import numpy as np
import tensorflow as tf

from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, graph=tf.Graph()):
        self.graph = graph

    @abc.abstractmethod
    def inference(self, inputs):
        pass


class MultiLayerPerceptron(Model):
    def __init__(self, input_size, output_size, hidden_layer_sizes,
                 graph=tf.Graph()):
        super(MultiLayerPerceptron, self).__init__(graph)
        self.input_size = input_size
        self.output_size = output_size
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, input_size])
            if self.output_size == 1:
                self.labels = tf.placeholder(tf.float32, [None])
            else:
                self.labels = tf.placeholder(tf.float32, [None, output_size])
        self.number_of_labels = output_size
        self.hidden_unit_sizes = hidden_layer_sizes

    def __str__(self):
        return 'MultiLayerPerceptron'

    def inference(self, inputs):
        hidden = inputs
        input_size = self.input_size
        for layer_index in range(len(self.hidden_unit_sizes)):
            output_size = self.hidden_unit_sizes[layer_index]
            with tf.name_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, output_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([output_size]), name='b')
                hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)
            input_size = output_size
        output_size = self.number_of_labels
        with tf.name_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            logits = tf.matmul(hidden, weights) + biases
        return tf.nn.log_softmax(logits)
