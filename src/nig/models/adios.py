from __future__ import absolute_import, division

import numpy as np
import tensorflow as tf

from ..learning.models import Model

__author__ = 'alshedivat'


class ADIOS(Model):
    """Architecture Deep In the Output Space.

    Composes an arbitrary input symbol with hierarchical multiple outputs.

    Arguments:
    ----------
        input_size : uint
        output_size : list
        hidden_layer_sizes : list
        activation : TF activation
        loss
        optimizer
        loss_summary
        grads_processor
    """
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 loss=None, loss_summary=False, optimizer=None,
                 optimizer_opts=None):
        assert len(output_size) == 2, "ADIOS works with exactly two outputs."
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        super(ADIOS, self).__init__(
            inputs=inputs, outputs=outputs, loss=loss,
            loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts)

    def _output_op(self, inputs):
        # Sanity check
        assert self.input_size == inputs.get_shape().dims[-1].value, \
            "Mismatch between the expected and actual input size."

        outputs = []

        # The first hidden and output layers are concatenated
        input_size = self.input_size
        output_size = self.output_size[0]
        with tf.variable_scope('output0_sigmoid_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            current = tf.sigmoid(tf.matmul(inputs, weights) + biases)
            outputs.append(current)
        if self.hidden_layer_sizes:
            hidden_size = self.hidden_layer_sizes[0]
            with tf.variable_scope('hidden0'):
                weights = tf.Variable(tf.random_normal(
                    [input_size, hidden_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([hidden_size]), name='b')
                hidden = self.activation(tf.matmul(inputs, weights) + biases)
                current = tf.concat(1, [current, hidden])

        # Add the rest of hidden layers
        input_size = current.get_shape().dims[-1].value
        for layer_index, hidden_size in enumerate(self.hidden_layer_sizes, 1):
            with tf.variable_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, hidden_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([hidden_size]), name='b')
                current = self.activation(tf.matmul(current, weights) + biases)
                input_size = hidden_size

        # Add the final output layer
        output_size = self.output_size[1]
        with tf.variable_scope('output1_sigmoid_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            current = tf.sigmoid(tf.matmul(current, weights) + biases)
            outputs.append(current)

        return outputs
