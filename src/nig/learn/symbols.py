import abc
import numpy as np
import tensorflow as tf

from nig.functions import PipelineFunction

__author__ = 'Emmanouil Antonios Platanios'


class Symbol(PipelineFunction):
    __metaclass__ = abc.ABCMeta

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
    def __init__(self, input_size, output_size, hidden_layer_sizes,
                 activation=tf.nn.sigmoid, softmax_output=True, use_log=True):
        super(MultiLayerPerceptron, self).__init__([input_size], [output_size])
        self.hidden_unit_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.use_log = use_log

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
                hidden = self.activation(tf.matmul(hidden, weights) + biases)
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
        if self.softmax_output and self.use_log:
            return tf.nn.log_softmax(outputs)
        elif self.softmax_output:
            return tf.nn.softmax(outputs)
        elif self.use_log:
            return tf.log(outputs)
        else:
            return outputs
