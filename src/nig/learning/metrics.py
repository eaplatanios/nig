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

__all__ = ['Metric', 'Accuracy', 'CrossEntropy', 'HammingLoss']


class Metric(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, name='metric'):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, prediction, truth):
        return self.evaluate(prediction, truth)

    @abc.abstractmethod
    def evaluate(self, prediction, truth):
        pass


class Accuracy(Metric):
    def __init__(self, one_hot_truth=False, name='accuracy'):
        super(Accuracy, self).__init__(name=name)
        self.one_hot_truth = one_hot_truth

    def evaluate(self, prediction, truth):
        with tf.name_scope(self.name):
            if self.one_hot_truth:
                metric = tf.equal(tf.argmax(prediction, 1), tf.argmax(truth, 1))
            else:
                truth = tf.to_int64(tf.squeeze(truth))
                metric = tf.nn.in_top_k(prediction, truth, 1)
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

    def evaluate(self, prediction, truth):
        with tf.name_scope(self.name):
            if not self.log_predictions:
                prediction = tf.log(prediction)
            if self.one_hot_truth:
                if self.scaled_predictions:
                    metric = truth * prediction
                    metric = -tf.reduce_sum(metric, reduction_indices=[1])
                else:
                    metric = tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction, labels=truth)
            else:
                # TODO: Make efficient for scaled predictions.
                truth = tf.to_int64(tf.squeeze(truth))
                metric = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction, labels=truth)
            metric = tf.reduce_mean(metric)
        return metric


class HammingLoss(Metric):
    def __init__(self, log_predictions=True, name='hamming_loss'):
        super(HammingLoss, self).__init__(name=name)
        self.log_predictions = log_predictions

    def evaluate(self, prediction, truth):
        with tf.name_scope(self.name):
            if self.log_predictions:
                prediction = tf.nn.relu(tf.sign(prediction - tf.log(0.5)))
            else:
                prediction = tf.nn.relu(tf.sign(prediction - 0.5))
            metric = tf.cast(tf.not_equal(prediction, truth), tf.float32)
            metric = tf.reduce_sum(metric, reduction_indices=[1])
            metric = tf.reduce_mean(metric)
            return metric
