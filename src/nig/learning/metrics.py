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


class Metric(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def __str__(self):
        pass

    def __call__(self, prediction, truth, name='metric'):
        return self.evaluate(prediction, truth, name)

    @abc.abstractmethod
    def evaluate(self, prediction, truth, name='metric'):
        pass


class CrossEntropyOneHotEncodingMetric(Metric):
    """Note that this needs probability quantities as predictions (i.e., having
    gone through a softmax layer)."""
    def __str__(self):
        return 'cross_entropy'

    def evaluate(self, prediction, truth, name='cross_entropy'):
        metric = -tf.reduce_sum(truth * prediction, reduction_indices=[1])
        metric = tf.reduce_mean(metric, name=name)
        return metric


class CrossEntropyIntegerEncodingMetric(Metric):
    """Note that this needs logit quantities as predictions (i.e., not having
    gone through a softmax layer)."""
    def __str__(self):
        return 'cross_entropy'

    def evaluate(self, prediction, truth, name='cross_entropy'):
        metric = tf.nn.sparse_softmax_cross_entropy_with_logits(
            prediction, tf.to_int64(tf.squeeze(truth)))
        metric = tf.reduce_mean(metric, name=name)
        return metric


class AccuracyOneHotEncodingMetric(Metric):
    """Note that this needs probability quantities as predictions (i.e., having
    gone through a softmax layer)."""
    def __str__(self):
        return 'accuracy'

    def evaluate(self, prediction, truth, name='accuracy'):
        metric = tf.equal(tf.argmax(prediction, 1), tf.argmax(truth, 1))
        metric = tf.reduce_mean(tf.cast(metric, tf.float32), name=name)
        return metric


class AccuracyIntegerEncodingMetric(Metric):
    """Note that this needs integer quantities as predictions (i.e., integer
    label corresponding to the most likely outcome)."""
    def __str__(self):
        return 'accuracy'

    def evaluate(self, prediction, truth, name='accuracy'):
        metric = tf.nn.in_top_k(
            prediction, tf.to_int64(tf.squeeze(truth)), 1)
        metric = tf.reduce_mean(tf.cast(metric, tf.float32), name=name)
        return metric
