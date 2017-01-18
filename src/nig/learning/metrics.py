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

from ..utilities.tensorflow import name_scope_context

__author__ = 'eaplatanios'

__all__ = [
    'Metric', 'CombinedMetric', 'L2Loss', 'Accuracy', 'Precision', 'Recall',
    'F1Score', 'HammingLoss', 'CrossEntropy']


class Metric(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, name='metric'):
        self.name = name
        self.name_scope = self.name.replace(' ', '_')

    def __str__(self):
        return self.name

    def __call__(self, outputs, train_outputs):
        return self.evaluate(outputs, train_outputs)

    @abc.abstractmethod
    def evaluate(self, outputs, train_outputs):
        pass


class CombinedMetric(Metric):
    def __init__(self, metrics, combination_function=None,
                 name='combined metric'):
        super(CombinedMetric, self).__init__(name=name)
        if combination_function is None:
            combination_function = lambda tensors: tf.add_n(tensors)
        self.combination_function = combination_function
        self.metrics = metrics

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        tensors = [metric(outputs, train_outputs) for metric in self.metrics]
        return self.combination_function(tensors)


class L2Loss(Metric):
    def __init__(self, name='l2 loss'):
        super(L2Loss, self).__init__(name=name)

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        metric = tf.square(tf.sub(outputs, train_outputs))
        num_samples = tf.cast(tf.shape(metric)[0], tf.float32)
        return tf.reduce_sum(metric) / num_samples


class _ClassificationMetric(Metric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='accuracy'):
        super(_ClassificationMetric, self).__init__(name=name)
        self.log_outputs = log_outputs
        self.scaled_outputs = scaled_outputs
        self.one_hot_train_outputs = one_hot_train_outputs
        self.thresholds = thresholds
        self.macro_average = macro_average

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        if not self.scaled_outputs:
            if self.log_outputs:
                outputs = tf.nn.log_softmax(outputs)
            else:
                outputs = tf.nn.softmax(outputs)
        if not self.one_hot_train_outputs:
            train_outputs = tf.one_hot(
                indices=train_outputs, depth=tf.shape(outputs)[1])
        if self.log_outputs:
            thresholds = tf.log(self.thresholds)
        else:
            thresholds = self.thresholds
        outputs = tf.nn.relu(tf.sign(outputs - thresholds))
        nominator, denominator = self._nominator_denominator(
            outputs=outputs, train_outputs=train_outputs)
        if self.macro_average:
            metric = tf.div(nominator, denominator)
            metric = tf.reduce_mean(metric, reduction_indices=[-1])
        else:
            nominator = tf.reduce_sum(nominator)
            denominator = tf.reduce_sum(denominator)
            metric = tf.div(nominator, denominator)
        return metric

    @abc.abstractmethod
    def _nominator_denominator(self, outputs, train_outputs):
        pass


class Accuracy(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='accuracy'):
        super(Accuracy, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _nominator_denominator(self, outputs, train_outputs):
        temp_outputs = tf.cast(tf.equal(outputs, train_outputs), tf.float32)
        nominator = tf.reduce_sum(temp_outputs, reduction_indices=[0])
        denominator = tf.shape(outputs)[0]
        return nominator, denominator


class Precision(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='precision'):
        super(Precision, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _nominator_denominator(self, outputs, train_outputs):
        true_positives = tf.select(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        nominator = tf.reduce_sum(true_positives, reduction_indices=[0])
        denominator = tf.reduce_sum(outputs, reduction_indices=[0])
        return nominator, denominator


class Recall(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='recall'):
        super(Recall, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _nominator_denominator(self, outputs, train_outputs):
        true_positives = tf.select(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        nominator = tf.reduce_sum(true_positives, reduction_indices=[0])
        denominator = tf.reduce_sum(train_outputs, reduction_indices=[0])
        return nominator, denominator


class F1Score(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='f1 score'):
        super(F1Score, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _nominator_denominator(self, outputs, train_outputs):
        true_positives = tf.select(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        all_positives = tf.reduce_sum(outputs, reduction_indices=[0])
        all_train_positives = tf.reduce_sum(
            train_outputs, reduction_indices=[0])
        precision = tf.div(true_positives, all_positives)
        recall = tf.div(true_positives, all_train_positives)
        nominator = tf.mul(2, tf.mul(precision, recall))
        denominator = tf.add(precision, recall)
        return nominator, denominator


class HammingLoss(Metric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 name='hamming loss'):
        super(HammingLoss, self).__init__(name=name)
        self.log_outputs = log_outputs
        self.scaled_outputs = scaled_outputs
        self.one_hot_train_outputs = one_hot_train_outputs
        self.thresholds = thresholds

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        if not self.scaled_outputs:
            if self.log_outputs:
                outputs = tf.nn.log_softmax(outputs)
            else:
                outputs = tf.nn.softmax(outputs)
        if not self.one_hot_train_outputs:
            train_outputs = tf.one_hot(
                indices=train_outputs, depth=tf.shape(outputs)[1])
        if self.log_outputs:
            thresholds = tf.log(self.thresholds)
        else:
            thresholds = self.thresholds
        outputs = tf.nn.relu(tf.sign(outputs - thresholds))
        metric = tf.cast(tf.not_equal(outputs, train_outputs), tf.float32)
        metric = tf.reduce_sum(metric, reduction_indices=[1])
        return tf.reduce_mean(metric)


class CrossEntropy(Metric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, name='cross entropy'):
        super(CrossEntropy, self).__init__(name=name)
        self.log_predictions = log_outputs
        self.scaled_predictions = scaled_outputs
        self.one_hot_train_outputs = one_hot_train_outputs

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        if not self.log_predictions:
            outputs = tf.log(outputs)
        if self.one_hot_train_outputs:
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
        return tf.reduce_mean(metric)
