from __future__ import absolute_import, division

import numpy as np
import tensorflow as tf

from ..learning.models import Model

__author__ = 'eaplatanios'


class MultiLayerPerceptron(Model):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 softmax_output=True, use_log=True, train_outputs_one_hot=False,
                 loss=None, loss_summary=False, optimizer=None,
                 optimizer_opts=None, grads_processor=None):
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.use_log = use_log
        self.train_outputs_one_hot = train_outputs_one_hot
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        train_outputs = None if train_outputs_one_hot \
            else tf.placeholder(tf.int32, shape=[None])
        super(MultiLayerPerceptron, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=train_outputs,
            loss=loss, loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts, grads_processor=grads_processor)

    def __str__(self):
        return 'MultiLayerPerceptron[{}:{}:{}:{}]'.format(
            self.inputs.get_shape()[1], self.output_size,
            self.hidden_layer_sizes, self.softmax_output)

    def _output_op(self, inputs):
        hidden = inputs
        input_size = inputs.get_shape().dims[-1].value
        for layer_index, output_size in enumerate(self.hidden_layer_sizes):
            with tf.variable_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, output_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
                biases = tf.Variable(tf.zeros([output_size]), name='b')
                hidden = self.activation(tf.matmul(hidden, weights) + biases)
            input_size = output_size
        with tf.variable_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, self.output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
            biases = tf.Variable(tf.zeros([self.output_size]), name='b')
            outputs = tf.matmul(hidden, weights) + biases
        if self.softmax_output and self.use_log:
            return tf.nn.log_softmax(outputs)
        elif self.softmax_output:
            return tf.nn.softmax(outputs)
        elif self.use_log:
            return tf.log(outputs)
        return outputs
