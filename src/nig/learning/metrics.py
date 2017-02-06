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
import numpy as np
import tensorflow as tf

from six import with_metaclass

from ..ops.classification_ops import accuracy
from ..utilities.tensorflow import name_scope_context

__author__ = 'eaplatanios'

__all__ = [
    'Metric', 'CombinedMetric', 'L2Loss', 'Accuracy', 'Precision', 'Recall',
    'F1Score', 'HammingLoss', 'CrossEntropy', 'BinaryCrossEntropy']

__EPS__ = np.finfo(np.float32).eps


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
        metric = tf.square(tf.subtract(outputs, train_outputs))
        return tf.reduce_sum(metric)


class Accuracy(Metric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='accuracy'):
        super(Accuracy, self).__init__(name=name)
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
        return accuracy(
            predictions=outputs, labels=train_outputs, thresholds=thresholds,
            weights=None, macro_average=self.macro_average,
            requested_ops='value')


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
        numerator, denominator = self._numerator_denominator(
            outputs=outputs, train_outputs=train_outputs)
        if self.macro_average:
            numerator = tf.where(
                tf.equal(denominator, 0.0),
                tf.fill(tf.shape(numerator), __EPS__),
                numerator)
            denominator = tf.where(
                tf.equal(denominator, 0.0),
                tf.fill(tf.shape(denominator), __EPS__),
                denominator)
            metric = tf.divide(numerator, denominator)
            metric = tf.reduce_mean(metric, axis=-1)
        else:
            numerator = tf.reduce_sum(numerator)
            denominator = tf.reduce_sum(denominator)
            numerator = tf.where(
                tf.equal(denominator, 0.0),
                tf.fill(tf.shape(numerator), __EPS__),
                numerator)
            denominator = tf.where(
                tf.equal(denominator, 0.0),
                tf.fill(tf.shape(denominator), __EPS__),
                denominator)
            metric = tf.divide(numerator, denominator)
        return metric

    @abc.abstractmethod
    def _numerator_denominator(self, outputs, train_outputs):
        pass


class Precision(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='precision'):
        super(Precision, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _numerator_denominator(self, outputs, train_outputs):
        true_positives = tf.where(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        numerator = tf.reduce_sum(true_positives, axis=0)
        denominator = tf.reduce_sum(outputs, axis=0)
        return numerator, denominator


class Recall(_ClassificationMetric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='recall'):
        super(Recall, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)

    def _numerator_denominator(self, outputs, train_outputs):
        true_positives = tf.where(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        numerator = tf.reduce_sum(true_positives, axis=0)
        denominator = tf.reduce_sum(train_outputs, axis=0)
        return numerator, denominator


class FScore(_ClassificationMetric):
    def __init__(self, beta=1.0, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name=None):
        if name is None:
            name = 'f-%.4f score' % beta
        super(FScore, self).__init__(
            log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)
        self.beta = beta

    def _numerator_denominator(self, outputs, train_outputs):
        true_positives = tf.where(
            tf.equal(outputs, 1.0),
            tf.cast(tf.equal(outputs, train_outputs), tf.float32), outputs)
        true_positives = tf.reduce_sum(true_positives, axis=0)
        output_positives = tf.reduce_sum(outputs, axis=0)
        total_positives = tf.reduce_sum(
            train_outputs, axis=0)
        true_positives_precision = tf.where(
            tf.equal(output_positives, 0.0),
            tf.fill(tf.shape(true_positives), __EPS__), true_positives)
        output_positives = tf.where(
            tf.equal(output_positives, 0.0),
            tf.fill(tf.shape(output_positives), __EPS__), output_positives)
        true_positives_recall = tf.where(
            tf.equal(total_positives, 0.0),
            tf.fill(tf.shape(true_positives), __EPS__), true_positives)
        total_positives = tf.where(
            tf.equal(total_positives, 0.0),
            tf.fill(tf.shape(total_positives), __EPS__), total_positives)
        precision = tf.divide(true_positives_precision, output_positives)
        recall = tf.divide(true_positives_recall, total_positives)
        beta_squared = self.beta * self.beta
        numerator = tf.multiply(
            1.0 + beta_squared, tf.multiply(precision, recall))
        denominator = tf.add(tf.multiply(beta_squared, precision), recall)
        return numerator, denominator


class F1Score(FScore):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, thresholds=0.5,
                 macro_average=True, name='f1 score'):
        super(F1Score, self).__init__(
            beta=1.0, log_outputs=log_outputs, scaled_outputs=scaled_outputs,
            one_hot_train_outputs=one_hot_train_outputs,
            thresholds=thresholds, macro_average=macro_average, name=name)


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
        metric = tf.reduce_sum(metric, axis=1)
        return tf.reduce_sum(metric)


class CrossEntropy(Metric):
    def __init__(self, log_outputs=True, scaled_outputs=False,
                 one_hot_train_outputs=False, name='cross entropy'):
        super(CrossEntropy, self).__init__(name=name)
        self.log_outputs = log_outputs
        self.scaled_outputs = scaled_outputs
        self.one_hot_train_outputs = one_hot_train_outputs

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        if not self.log_outputs:
            outputs = tf.log(outputs)
        if self.one_hot_train_outputs:
            if self.scaled_outputs:
                metric = -tf.reduce_sum(train_outputs * outputs, axis=1)
            else:
                metric = tf.nn.softmax_cross_entropy_with_logits(
                    logits=outputs, labels=train_outputs)
        else:
            # TODO: Make efficient for scaled predictions.
            train_outputs = tf.to_int64(tf.squeeze(train_outputs))
            metric = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, labels=train_outputs)
        return tf.reduce_sum(metric)


class BinaryCrossEntropy(Metric):
    def __init__(self, logit_outputs=True, one_hot_train_outputs=False,
                 name='binary cross entropy'):
        super(BinaryCrossEntropy, self).__init__(name=name)
        self.logit_outputs = logit_outputs
        self.one_hot_train_outputs = one_hot_train_outputs
        # self.numerically_robust = numerically_robust

    @name_scope_context
    def evaluate(self, outputs, train_outputs):
        if not self.logit_outputs:
            epsilon = tf.convert_to_tensor(
                __EPS__, dtype=outputs.dtype.base_dtype)
            outputs = tf.clip_by_value(outputs, epsilon, 1.0 - epsilon)
            outputs = tf.log(tf.div(outputs, 1.0 - outputs))
        if not self.one_hot_train_outputs:
            train_outputs = tf.to_int64(tf.squeeze(train_outputs))
            train_outputs = tf.one_hot(
                indices=train_outputs, depth=tf.shape(outputs)[1], axis=-1)
        metric = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=train_outputs, logits=outputs)
        return tf.reduce_sum(metric)
