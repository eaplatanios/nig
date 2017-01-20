# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from ..learning.models import Model

__author__ = 'eaplatanios'

__all__ = ['LinearCombination', 'MultiLayerPerceptron']


class LinearCombination(Model):
    def __init__(self, inputs, axis, convex=True, loss=None, loss_summary=False,
                 optimizer=None, optimizer_opts=None):
        self.axis = axis
        self.convex = convex
        if isinstance(inputs, tf.Tensor):
            _inputs_shape = inputs.get_shape().as_list()
        elif not isinstance(inputs, list):
            _inputs_shape = list(inputs)
        else:
            _inputs_shape = inputs
        self._weights_shape = _inputs_shape.copy()
        _inputs_shape.pop(self.axis)
        _train_outputs_shape = _inputs_shape
        for axis in range(len(self._weights_shape)):
            if axis != self.axis:
                self._weights_shape[axis] = 1
        with tf.name_scope('linear'):
            if not isinstance(inputs, tf.Tensor):
                inputs = tf.placeholder(
                    tf.float32, shape=inputs.get_shape(), name='inputs')
            outputs = self._output_op(inputs)
            train_outputs = tf.placeholder(
                tf.float32, shape=_train_outputs_shape, name='train_outputs')
        super(LinearCombination, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=train_outputs,
            loss=loss, loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts)

    def __str__(self):
        return 'LinearCombination[{}]'.format(self.inputs.get_shape()[1])

    def _output_op(self, inputs):
        weights = tf.Variable(tf.ones(self._weights_shape), name='weight')
        if self.convex:
            weights = tf.square(weights)
            weights = tf.divide(weights, tf.reduce_sum(weights))
        outputs = tf.multiply(inputs, weights)
        outputs = tf.reduce_sum(outputs, axis=self.axis)
        return outputs


class MultiLayerPerceptron(Model):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 softmax_output=True, sigmoid_output=False, log_output=True,
                 train_outputs_one_hot=False, loss=None, loss_summary=False,
                 optimizer=None, optimizer_opts=None):
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.sigmoid_output = sigmoid_output
        self.log_output = log_output
        self.train_outputs_one_hot = train_outputs_one_hot
        with tf.name_scope('multi_layer_perceptron'):
            inputs = tf.placeholder(
                tf.float32, shape=[None, input_size], name='inputs')
            outputs = self._output_op(inputs)
            if self.train_outputs_one_hot:
                train_outputs = tf.placeholder(
                    tf.float32, shape=[None, output_size], name='train_outputs')
            else:
                train_outputs = tf.placeholder(
                    tf.int32, shape=[None], name='train_outputs')
        super(MultiLayerPerceptron, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=train_outputs,
            loss=loss, loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts)

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
        if self.softmax_output and self.log_output:
            return tf.nn.log_softmax(outputs)
        elif self.softmax_output:
            return tf.nn.softmax(outputs)
        elif self.sigmoid_output and self.log_output:
            return tf.log(tf.sigmoid(outputs))
        elif self.log_output:
            return tf.log(outputs)
        elif self.sigmoid_output:
            return tf.sigmoid(outputs)
        return outputs
