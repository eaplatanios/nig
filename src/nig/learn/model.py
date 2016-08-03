import abc
import numpy as np
import tensorflow as tf
import os

from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


class Model(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def inference(self, inputs):
        pass


class MultiLayerPerceptron(Model):
    def __init__(self, input_shape, number_of_labels, hidden_unit_sizes, learning_rate,
                 maximum_number_of_iterations=100000, working_directory=os.getcwd(), graph=tf.Graph()):
        self.input_shape = input_shape if type(input_shape) is list else [input_shape]
        self.output_shape = [number_of_labels]
        self.learning_rate = learning_rate
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.working_directory = working_directory
        self.graph = graph
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None] + self.input_shape)
            if self.output_shape == [1]:
                self.labels = tf.placeholder(tf.float32, [None])
            else:
                self.labels = tf.placeholder(tf.float32, [None] + self.output_shape)
        self.number_of_labels = number_of_labels
        self.hidden_unit_sizes = hidden_unit_sizes

    def __str__(self):
        return 'MultiLayerPerceptron'

    def inference(self, inputs):
        hidden = tf.contrib.layers.flatten(inputs)
        input_size = tf.shape(hidden)[1]
        for layer_index in range(len(self.hidden_unit_sizes)):
            output_size = self.hidden_unit_sizes[layer_index]
            with tf.name_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal([input_size, output_size],
                                                       stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
                biases = tf.Variable(tf.zeros([output_size]), name='b')
                hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)
            input_size = output_size
        output_size = self.number_of_labels
        with tf.name_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal([input_size, output_size],
                                                   stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            logits = tf.matmul(hidden, weights) + biases
        return tf.nn.log_softmax(logits)
