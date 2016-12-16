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

import abc
import tensorflow as tf

from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = [
    'Metric', 'CombinedMetric', 'L2Loss', 'Accuracy', 'CrossEntropy',
    'HammingLoss']


class Metric(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, name='metric'):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, outputs, train_outputs):
        return self.evaluate(outputs, train_outputs)

    @abc.abstractmethod
    def evaluate(self, outputs, train_outputs):
        pass


class CombinedMetric(Metric):
    def __init__(self, metrics, combination_function=None,
                 name='combined_metric'):
        super(CombinedMetric, self).__init__(name=name)
        if combination_function is None:
            combination_function = lambda tensors: tf.add_n(tensors)
        self.combination_function = combination_function
        self.metrics = metrics

    def evaluate(self, outputs, train_outputs):
        tensors = [metric(outputs, train_outputs) for metric in self.metrics]
        return self.combination_function(tensors)


class L2Loss(Metric):
    def __init__(self, name='l2_loss'):
        super(L2Loss, self).__init__(name=name)

    def evaluate(self, outputs, train_outputs):
        with tf.name_scope(self.name):
            metric = tf.square(tf.sub(outputs - train_outputs))
            metric = tf.reduce_sum(metric)
        return metric


class Accuracy(Metric):
    def __init__(self, one_hot_truth=False, name='accuracy'):
        super(Accuracy, self).__init__(name=name)
        self.one_hot_truth = one_hot_truth

    def evaluate(self, outputs, train_outputs):
        with tf.name_scope(self.name):
            if self.one_hot_truth:
                outputs = tf.argmax(outputs, 1)
                train_outputs = tf.argmax(train_outputs, 1)
                metric = tf.equal(outputs, train_outputs)
            else:
                train_outputs = tf.to_int64(tf.squeeze(train_outputs))
                metric = tf.nn.in_top_k(outputs, train_outputs, 1)
            metric = tf.cast(metric, tf.float32)
            metric = tf.reduce_mean(metric)
        return metric


class CrossEntropy(Metric):
    def __init__(self, log_predictions=True, scaled_predictions=False,
                 one_hot_truth=False, name='cross_entropy'):
        super(CrossEntropy, self).__init__(name=name)
        self.log_predictions = log_predictions
        self.scaled_predictions = scaled_predictions
        self.one_hot_truth = one_hot_truth

    def evaluate(self, outputs, train_outputs):
        with tf.name_scope(self.name):
            if not self.log_predictions:
                outputs = tf.log(outputs)
            if self.one_hot_truth:
                if self.scaled_predictions:
                    metric = train_outputs * outputs
                    metric = -tf.reduce_sum(metric, reduction_indices=[1])
                else:
                    metric = tf.nn.softmax_cross_entropy_with_logits(
                        logits=outputs, labels=train_outputs)
            else:
                # TODO: Make efficient for scaled predictions.
                train_outputs = tf.to_int64(tf.squeeze(train_outputs))
                metric = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=outputs, labels=train_outputs)
            metric = tf.reduce_mean(metric)
        return metric


class HammingLoss(Metric):
    def __init__(self, log_predictions=True, name='hamming_loss'):
        super(HammingLoss, self).__init__(name=name)
        self.log_predictions = log_predictions

    def evaluate(self, outputs, train_outputs):
        with tf.name_scope(self.name):
            if self.log_predictions:
                outputs = tf.nn.relu(tf.sign(outputs - tf.log(0.5)))
            else:
                outputs = tf.nn.relu(tf.sign(outputs - 0.5))
            metric = tf.cast(tf.not_equal(outputs, train_outputs), tf.float32)
            metric = tf.reduce_sum(metric, reduction_indices=[1])
            metric = tf.reduce_mean(metric)
            return metric
