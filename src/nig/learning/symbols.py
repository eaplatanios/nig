import abc
import numpy as np
import tensorflow as tf
from six import with_metaclass

from nig.utilities.functions import PipelineFunction

__author__ = 'eaplatanios'


class Symbol(with_metaclass(abc.ABCMeta, PipelineFunction)):
    def __init__(self, input_shape, output_shape):
        super(Symbol, self).__init__(self.op, min_num_args=1)
        assert isinstance(input_shape, list)
        assert isinstance(output_shape, list)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, inputs):
        return self.op(inputs)

    @abc.abstractmethod
    def op(self, inputs):
        pass


class Input(Symbol):
    def __init__(self, shape):
        super(Input, self).__init__(None, shape)

    def __str__(self):
        return 'Input'

    def op(self, inputs):
        return inputs


class MultiLayerPerceptron(Symbol):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 softmax_output=True, use_log=True):
        super(MultiLayerPerceptron, self).__init__([input_size], [output_size])
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.use_log = use_log

    def __str__(self):
        return 'MultiLayerPerceptron[{}:{}:{}:{}]' \
            .format(self.input_shape[0], self.output_shape[0],
                    self.hidden_layer_sizes, self.softmax_output)

    def op(self, inputs):
        hidden = inputs
        input_size = self.input_shape[0]
        for layer_index, output_size in enumerate(self.hidden_layer_sizes):
            with tf.variable_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, output_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([output_size]), name='b')
                hidden = self.activation(tf.matmul(hidden, weights) + biases)
            input_size = output_size
        output_size = self.output_shape[0]
        with tf.variable_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            outputs = tf.matmul(hidden, weights) + biases
        if self.softmax_output and self.use_log:
            return tf.nn.log_softmax(outputs)
        elif self.softmax_output:
            return tf.nn.softmax(outputs)
        elif self.use_log:
            return tf.log(outputs)
        else:
            return outputs


class ADIOS(Symbol):
    """Architectures Deep In the Output Space (ADIOS).

    Composes an arbitrary input symbol with hierarchical multiple outputs.

    Arguments:
    ----------
        hidden_layer_sizes : list
        output_sizes : list
        activation : TF activation
    """
    def __init__(self, input_sizes, output_sizes, hidden_layer_sizes,
                 activation):
        assert len(output_sizes) == 2, "ADIOS works with exactly two outputs."
        super(ADIOS, self).__init__(input_sizes, output_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def op(self, inputs):
        outputs = []

        # The first hidden and output layers are concatenated
        input_size = inputs.get_shape().dims[-1].value
        output_size = self.output_shape[0]
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
        output_size = self.output_shape[1]
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
