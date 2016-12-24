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
import os
import tensorflow as tf

from ..learning import learners
from ..models import rbm

__author__ = 'eaplatanios'

__all__ = ['RBMIntegrator']


class RBMIntegrator(object):
    def __init__(self, num_functions, estimate_errors=False,
                 working_dir=os.getcwd(), persistent=False):
        self.num_functions = num_functions
        self.estimate_errors = estimate_errors
        self.working_dir = working_dir
        self.persistent = persistent
        self._learners = None
        self._num_labels = None
        self._has_run = False
        self._errors = None

    def run(self, labeled_predictions, labeled_truth, unlabeled_predictions):
        if self._learners is None:
            self._num_labels = unlabeled_predictions.shape[2]
            optimizer = lambda: tf.train.AdamOptimizer()
            optimizer_opts = {
                'batch_size': 100, 'max_iter': 10000, 'abs_loss_chg_tol': 1e-6,
                'rel_loss_chg_tol': 1e-6, 'loss_chg_iter_below_tol': 5,
                'grads_processor': None}
            self._learners = []
            for label in range(self._num_labels):
                with tf.Graph().as_default():
                    inputs = tf.placeholder(
                        tf.float32, shape=[None, self.num_functions],
                        name='input')
                    model = rbm.RBM(
                        inputs=inputs, num_hidden=1, mean_field=True,
                        mean_field_cd=False, cd_steps=1, loss_summary=False,
                        optimizer=optimizer, optimizer_opts=optimizer_opts,
                        graph=None)
                    if self.estimate_errors:
                        integrated = tf.tile(
                            input=model.outputs,
                            multiples=[1, self.num_functions])
                        e = tf.abs(tf.sub(integrated, inputs))
                        e = tf.reduce_mean(e, reduction_indices=[0])
                        predict_postprocess = lambda o, e_op=e: [o, e_op]
                    else:
                        predict_postprocess = None
                l = learners.SimpleLearner(
                    model=model, new_graph=False,
                    predict_postprocess=predict_postprocess, logging_level=0)
                self._learners.append(l)
        else:
            # TODO: Add error messages.
            if labeled_predictions is not None:
                assert labeled_predictions.shape[2] == self._num_labels
            if labeled_truth is not None:
                assert labeled_truth.shape[1] == self._num_labels
            if unlabeled_predictions is not None:
                assert unlabeled_predictions.shape[2] == self._num_labels
        if self._has_run:
            init_option = -1 if self.persistent else False
        else:
            init_option = True
            self._has_run = False
        save_trained = self.persistent
        integrated = []
        errors = []
        for label in range(self._num_labels):
            working_dir = os.path.join(self.working_dir, 'label_%d' % label)
            data = np.squeeze(unlabeled_predictions[:, :, label])
            learner = self._learners[label]
            learner.train(
                data=data, init_option=init_option, callbacks=[],
                working_dir=working_dir, save_trained=save_trained)
            predictions = learner.predict(
                data=data, ckpt=None, working_dir=working_dir)
            if np.mean(predictions[1]) > 0.5:
                predictions[0] = 1 - predictions[0]
                predictions[1] = 1 - predictions[1]
            integrated.append(predictions[0])
            errors.append(predictions[1][None, :])
        integrated = np.concatenate(integrated, axis=1)  # B x O
        errors = np.concatenate(errors, axis=0)  # O x M
        return integrated, errors


class SemiSupervisedRBMIntegrator(object):
    def __init__(self, num_functions, estimate_errors=False,
                 working_dir=os.getcwd(), persistent=False):
        self.num_functions = num_functions
        self.estimate_errors = estimate_errors
        self.working_dir = working_dir
        self.persistent = persistent
        self._learners = None
        self._num_labels = None
        self._has_run = False
        self._errors = None

    def run(self, labeled_predictions, labeled_truth, unlabeled_predictions):
        if self._learners is None:
            self._num_labels = unlabeled_predictions.shape[2]
            optimizer = lambda: tf.train.AdamOptimizer()
            optimizer_opts = {
                'batch_size': None, 'max_iter': 10000, 'abs_loss_chg_tol': 1e-6,
                'rel_loss_chg_tol': 1e-6, 'loss_chg_iter_below_tol': 5,
                'grads_processor': None}
            self._learners = []
            for label in range(self._num_labels):
                with tf.Graph().as_default():
                    inputs = tf.placeholder(
                        tf.float32, shape=[None, self.num_functions],
                        name='input')
                    model = rbm.SemiSupervisedRBM(
                        inputs=inputs, num_hidden=1, persistent=True,
                        mean_field=True, mean_field_cd=False, cd_steps=1,
                        loss_summary=False, optimizer=optimizer,
                        optimizer_opts=optimizer_opts, graph=None)
                    if self.estimate_errors:
                        integrated = tf.tile(
                            input=model.outputs,
                            multiples=[1, self.num_functions])
                        e = tf.abs(tf.sub(integrated, inputs))
                        e = tf.reduce_mean(e, reduction_indices=[0])
                        predict_postprocess = lambda o, e_op=e: [o, e_op]
                    else:
                        predict_postprocess = None
                l = learners.SimpleLearner(
                    model=model, new_graph=False,
                    predict_postprocess=predict_postprocess, logging_level=0)
                self._learners.append(l)
        else:
            # TODO: Add error messages.
            if labeled_predictions is not None:
                assert labeled_predictions.shape[2] == self._num_labels
            if labeled_truth is not None:
                assert labeled_truth.shape[1] == self._num_labels
            if unlabeled_predictions is not None:
                assert unlabeled_predictions.shape[2] == self._num_labels
        if self._has_run:
            init_option = -1 if self.persistent else False
        else:
            init_option = True
            self._has_run = False
        save_trained = self.persistent
        integrated = []
        errors = []
        for label in range(self._num_labels):
            working_dir = os.path.join(self.working_dir, 'label_%d' % label)
            labeled_data = np.squeeze(labeled_predictions[:, :, label])
            labeled_labels = labeled_truth[:, label]
            unlabeled_data = np.squeeze(unlabeled_predictions[:, :, label])
            unlabeled_labels = -np.ones(len(unlabeled_data))
            data = np.concatenate((labeled_data, unlabeled_data), axis=0)
            truth = np.concatenate((labeled_labels, unlabeled_labels), axis=0)
            indices = np.random.permutation(np.arange(len(data)))
            data = data[indices], truth[indices]
            learner = self._learners[label]
            learner.train(
                data=data, init_option=init_option, callbacks=[],
                working_dir=working_dir, save_trained=save_trained)
            predictions = learner.predict(
                data=unlabeled_data, ckpt=None, working_dir=working_dir)
            if self.estimate_errors:
                if np.mean(predictions[1]) > 0.5:
                    predictions[0] = 1 - predictions[0]
                    predictions[1] = 1 - predictions[1]
                integrated.append(predictions[0])
                errors.append(predictions[1][None, :])
            else:
                integrated.append(predictions)
        integrated = np.concatenate(integrated, axis=1)  # B x O
        if self.estimate_errors:
            errors = np.concatenate(errors, axis=0)  # O x M
            return integrated, errors
        return integrated
