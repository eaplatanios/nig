import abc
import numpy as np
import tensorflow as tf

from nig.functions import pipe

__author__ = 'Emmanouil Antonios Platanios'


class Symbol(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape, output_shape):
        assert isinstance(input_shape, list)
        assert isinstance(output_shape, list)
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abc.abstractmethod
    def op(self, inputs):
        pass

    def add(self, symbol):
        return StackedSymbol(self, symbol)


class StackedSymbol(Symbol):
    def __init__(self, *symbols):
        super(StackedSymbol, self).__init__(symbols[0].input_shape,
                                            symbols[-1].output_shape)
        self.symbols = symbols

    def op(self, inputs):
        return pipe([symbol.op for symbol in self.symbols])(inputs)


class Input(Symbol):
    def __init__(self, shape):
        super(Input, self).__init__(None, shape)

    def __str__(self):
        return 'Input'

    def op(self, inputs):
        return inputs


class MultiLayerPerceptron(Symbol):
    def __init__(self, input_size, output_size, hidden_layer_sizes,
                 softmax_output=True):
        super(MultiLayerPerceptron, self).__init__([input_size], [output_size])
        self.hidden_unit_sizes = hidden_layer_sizes
        self.softmax_output = softmax_output

    def __str__(self):
        return 'MultiLayerPerceptron[{}:{}:{}:{}]' \
            .format(self.input_shape[0], self.output_shape[0],
                    self.hidden_unit_sizes, self.softmax_output)

    def op(self, inputs):
        hidden = inputs
        input_size = self.input_shape[0]
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
        output_size = self.output_shape[0]
        with tf.name_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            outputs = tf.matmul(hidden, weights) + biases
        return tf.nn.log_softmax(outputs) if self.softmax_output else outputs
