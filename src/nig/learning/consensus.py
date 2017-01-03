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
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from six import with_metaclass

from . import learners, metrics
from .models import CombinedModel
from ..models import rbm
from ..utilities.tensorflow import graph_context
from ..utilities.functions import memoize

__author__ = 'eaplatanios'

__all__ = ['Consensus', 'MajorityVote', 'HardMajorityVote',
           'ArgMaxMajorityVote', 'RBMConsensus', 'ConsensusLearnerLoss',
           'ConsensusLearner']

logger = logging.getLogger(__name__)


def _get_feed_dict(models, labeled_data=None, unlabeled_data=None):
    feed_dict = dict()
    for model in models:
        if labeled_data is not None:
            model_feed_dict = model.get_feed_dict(
                data=labeled_data, is_train=True)
        else:
            model_feed_dict = None
        if unlabeled_data is not None:
            model_feed_dict = model.get_feed_dict(
                data=unlabeled_data, is_train=False,
                feed_dict=model_feed_dict)
        feed_dict.update(model_feed_dict)
    return feed_dict


class Consensus(with_metaclass(abc.ABCMeta, object)):
    def __call__(self, models):
        return self._combine_outputs(models=models)

    @abc.abstractmethod
    def _combine_outputs(self, models):
        """
        Args:
            outputs (tf.Tensor): Tensor containing the outputs from multiple
                models, with dimensions `[B x M x ...]`, where `B` is the batch
                size, `M` is the number of models, and `...` corresponds to any
                number of other dimensions.
        """
        pass

    def train(self, state, session, labeled_data=None, unlabeled_data=None,
              max_iter=100, abs_loss_chg_tol=1e-3, rel_loss_chg_tol=1e-3,
              loss_chg_iter_below_tol=5):
        """

        Args:
            state:
            session:
            labeled_data:
            unlabeled_data:
            max_iter:
            abs_loss_chg_tol:
            rel_loss_chg_tol:
            loss_chg_iter_below_tol:

        Note:
            This method only needs to be implemented for trainable consensus
            methods.

        Returns:

        """
        pass


class MajorityVote(Consensus):
    def __init__(self, log_outputs=False, log_consensus=False):
        self.log_outputs = log_outputs
        self.log_consensus = log_consensus

    def _combine_outputs(self, models):
        outputs = [model.outputs for model in models]
        outputs = tf.pack(outputs, axis=1)  # B x M x O
        outputs = tf.exp(outputs) if self.log_outputs else outputs
        consensus = tf.reduce_mean(outputs, reduction_indices=[1])
        if self.log_consensus:
            consensus = tf.log(consensus)
        return consensus, None


class HardMajorityVote(Consensus):
    def __init__(self, log_outputs=False):
        self.log_outputs = log_outputs

    def _combine_outputs(self, models):
        outputs = [model.outputs for model in models]
        outputs = tf.pack(outputs, axis=1)  # B x M x O
        outputs = tf.exp(outputs) if self.log_outputs else outputs
        consensus = tf.reduce_mean(outputs, reduction_indices=[1])
        consensus = tf.select(
            consensus >= 0.5, tf.ones_like(consensus), tf.zeros_like(consensus))
        return consensus, None


class ArgMaxMajorityVote(Consensus):
    """
    Note:
        The model outputs in this case need to have dimensions `[B x M x O]`,
        where `B` is the batch size, `M` is the number of models, and `O` is
        the number of labels.
    """
    def __init__(self, log_outputs=False):
        self.log_outputs = log_outputs

    def _combine_outputs(self, models):
        outputs = [model.outputs for model in models]
        outputs = tf.pack(outputs, axis=1)  # B x M x O
        outputs = tf.exp(outputs) if self.log_outputs else outputs
        consensus = tf.reduce_mean(outputs, reduction_indices=[1])
        consensus = tf.argmax(consensus, axis=1)
        consensus = tf.one_hot(consensus, tf.shape(outputs)[2], axis=1)
        return consensus, None


class RBMConsensus(Consensus):
    """
    Note:
        The model outputs in this case need to have dimensions `[B x M x O]`,
        where `B` is the batch size, `M` is the number of models, and `O` is
        the number of labels.
    """
    def __init__(self, log_outputs=False, log_consensus=False):
        self.log_outputs = log_outputs
        self.log_consensus = log_consensus

    def _combine_outputs(self, models):
        outputs = [model.outputs for model in models]
        outputs = tf.pack(outputs, axis=1)  # B x M x O
        outputs = tf.exp(outputs) if self.log_outputs else outputs
        outputs = tf.unpack(outputs, axis=2)  # O x [B x M]
        optimizer = lambda: tf.train.AdamOptimizer()
        optimizer_opts = {
            'abs_loss_chg_tol': 1e-3, 'rel_loss_chg_tol': 1e-3,
            'loss_chg_iter_below_tol': 5, 'grads_processor': None}
        integrators = [
            rbm.SemiSupervisedRBM(
                inputs=tf.stop_gradient(output),
                num_hidden=1, persistent=False, mean_field=True,
                mean_field_cd=False, cd_steps=1, loss_summary=False,
                optimizer=optimizer, optimizer_opts=optimizer_opts,
                graph=None)
            for output in outputs]
        consensus = [i.outputs for i in integrators]
        consensus = tf.squeeze(tf.pack(consensus, axis=1))
        if self.log_consensus:
            consensus = tf.log(consensus)
        state = (models, integrators)
        return consensus, state

    def train(self, state, session, labeled_data=None, unlabeled_data=None,
              max_iter=100, abs_loss_chg_tol=1e-3, rel_loss_chg_tol=1e-3,
              loss_chg_iter_below_tol=5):
        models, integrators = state
        untrained_integrators = list(range(len(integrators)))
        prev_loss = [sys.float_info.max] * len(integrators)
        iter_below_tol = [0] * len(integrators)
        step = 0
        while True:
            fetches = [[integrators[i].train_op,
                        integrators[i].loss_op]
                       for i in untrained_integrators]
            feed_dict = dict()
            train_outputs = None
            if labeled_data is not None:
                labeled_data_batch = labeled_data.next()
                feed_dict = _get_feed_dict(
                    models=models, labeled_data=labeled_data_batch)
                train_outputs = [labeled_data_batch[-1][:, i]
                                 for i in untrained_integrators]
            if unlabeled_data is not None:
                unlabeled_data_batch = unlabeled_data.next()
                unlabeled_feed_dict = _get_feed_dict(
                    models=models, unlabeled_data=unlabeled_data_batch)
                if train_outputs is not None:
                    for k in unlabeled_feed_dict.keys():
                        feed_dict[k] = np.concatenate(
                            (feed_dict[k], unlabeled_feed_dict[k]), axis=0)
                    missing_labels = -np.ones(unlabeled_data.batch_size)
                    train_outputs = [np.concatenate((o, missing_labels), axis=0)
                                     for o in train_outputs]
                else:
                    feed_dict = unlabeled_feed_dict
            if train_outputs is not None:
                feed_dict.update(
                    {integrators[i].train_outputs: o
                     for i, o in enumerate(train_outputs)})
            run_outputs = session.run(fetches=fetches, feed_dict=feed_dict)
            for i_index, i in enumerate(untrained_integrators[:]):
                loss = run_outputs[i_index][1]
                # if i_index == 0 and step % 10 == 0:
                #     logger.info('Loss 0: %11.4e', loss)
                loss_diff = abs(prev_loss[i] - loss)
                if loss_diff < abs_loss_chg_tol \
                        or abs(loss_diff / prev_loss[i]) < rel_loss_chg_tol:
                    iter_below_tol[i] += 1
                else:
                    iter_below_tol[i] = 0
                if iter_below_tol[i] >= loss_chg_iter_below_tol:
                    # logger.info('Integrator %d finished training: Loss value '
                    #             'converged.' % i)
                    untrained_integrators.remove(i)
                elif step >= max_iter - 1:
                    # logger.info('Integrator %d finished training: Maximum '
                    #             'number of iterations reached.' % i)
                    untrained_integrators.remove(i)
                prev_loss[i] = loss
            if len(untrained_integrators) == 0:
                # logger.info('All integrators have finished training.')
                break
            step += 1


class ConsensusLearnerLoss(metrics.Metric):
    def __init__(self, model, consensus, consensus_loss_weight=1.0,
                 consensus_loss_metric=None, name='combined_loss'):
        super(ConsensusLearnerLoss, self).__init__(name=name)
        self.model_loss = model.loss
        self.consensus = consensus
        self.consensus_loss_weight = consensus_loss_weight
        self.consensus_loss_metric = consensus_loss_metric

    def evaluate(self, outputs, train_outputs):
        num_labeled = tf.shape(train_outputs)[0]
        model_loss = self.model_loss(outputs[:num_labeled], train_outputs)
        # model_loss = tf.mul(1-self.consensus_loss_weight, model_loss)
        if self.consensus_loss_metric is None:
            consensus_loss = self.model_loss(
                outputs[num_labeled:], self.consensus[num_labeled:])
        else:
            consensus_loss = self.consensus_loss_metric(
                outputs[num_labeled:], self.consensus[num_labeled:])
        consensus_loss = tf.mul(self.consensus_loss_weight, consensus_loss)
        # model_loss = tf.Print(model_loss, [model_loss], 'Model Loss: ', first_n=1000, summarize=1000)
        # consensus_loss = tf.Print(consensus_loss, [consensus_loss], 'Consensus Loss: ', first_n=1000, summarize=1000)
        return tf.add(model_loss, consensus_loss)


class ConsensusLearner(learners.Learner):
    def __init__(self, models, consensus_method=RBMConsensus(),
                 consensus_loss_weight=1.0, consensus_loss_metric=None,
                 first_consensus=10, first_consensus_max_iter=10000,
                 consensus_update_frequency=10, consensus_update_max_iter=500,
                 new_graph=False, session=None, predict_postprocess=None,
                 logging_level=0):
        if any(model.uses_external_optimizer for model in models):
            raise ValueError('Only internal optimizers are supported for this '
                             'learner.')
        self.consensus_loss_metric = consensus_loss_metric
        self.first_consensus = first_consensus
        self.first_consensus_max_iter = first_consensus_max_iter
        self.consensus_update_frequency = consensus_update_frequency
        self.consensus_update_max_iter = consensus_update_max_iter
        super(ConsensusLearner, self).__init__(
            models=models, new_graph=new_graph, session=session,
            predict_postprocess=predict_postprocess,
            logging_level=logging_level)
        if self.trainable:
            if not isinstance(consensus_method, Consensus):
                raise TypeError('Invalid consensus method type %s.'
                                % type(consensus_method))
            self.consensus_method = consensus_method
            with self.graph.as_default(), tf.name_scope('consensus_learner'):
                self.consensus_loss_weight = consensus_loss_weight
                self._consensus_loss_weight_var = tf.Variable(
                    initial_value=0.0, trainable=False, dtype=tf.float32)
                self.consensus, self.consensus_state = \
                    self.consensus_method(self.models)
                self.consensus = tf.stop_gradient(self.consensus)
                for m, model in enumerate(self.models):
                    loss = ConsensusLearnerLoss(
                        model=model, consensus=self.consensus,
                        consensus_loss_weight=self._consensus_loss_weight_var,
                        consensus_loss_metric=self.consensus_loss_metric)
                    model.update_loss(loss)
        else:
            raise ValueError('ConsensusLearner can only be used with trainable '
                             'models.')

    def copy(self, new_graph=False):
        return ConsensusLearner(
            models=self.models,
            consensus_loss_weight=self.consensus_loss_weight,
            consensus_method=self.consensus_method,
            consensus_loss_metric=self.consensus_loss_metric,
            first_consensus=self.first_consensus,
            first_consensus_max_iter=self.first_consensus_max_iter,
            consensus_update_frequency=self.consensus_update_frequency,
            consensus_update_max_iter=self.consensus_update_max_iter,
            new_graph=new_graph, session=self.initial_session,
            predict_postprocess=self.predict_postprocess,
            logging_level=self.logging_level)

    @graph_context
    def train(self, labeled_data, pipelines=None, init_option=-1,
              per_model_callbacks=None, combined_model_callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False,
              unlabeled_data=None, unlabeled_pipelines=None):
        if not self.trainable:
            raise ValueError('At least one of the models is not trainable.')
        models = self.models
        batch_size = [model.optimizer_opts.get('batch_size', None)
                      for model in models]
        if any(size != batch_size[0] for size in batch_size):
            raise ValueError('All batch sizes must match.')
        batch_size = batch_size[0]
        max_iter = [model.optimizer_opts.get('max_iter', 10000)
                    for model in models]
        abs_loss_chg_tol = [model.optimizer_opts.get('abs_loss_chg_tol', 1e-10)
                            for model in models]
        rel_loss_chg_tol = [model.optimizer_opts.get('rel_loss_chg_tol', 1e-3)
                            for model in models]
        loss_chg_iter_below_tol = [
            model.optimizer_opts.get('loss_chg_iter_below_tol', 5)
            for model in models]
        labeled_data = learners.get_iterator(
            data=labeled_data, batch_size=batch_size, cycle=True,
            pipelines=pipelines)
        if unlabeled_data is not None:
            if unlabeled_pipelines is None:
                unlabeled_pipelines = pipelines
            unlabeled_data = learners.get_iterator(
                data=unlabeled_data, cycle=True, pipelines=unlabeled_pipelines)
        per_model_callbacks = [learners.process_callbacks(per_model_callbacks)
                               for _ in self.models]
        combined_model_callbacks = \
            learners.process_callbacks(combined_model_callbacks)
        summary_writer = tf.summary.FileWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        feed_dict = _get_feed_dict(
            models=self.models, labeled_data=labeled_data.next(),
            unlabeled_data=unlabeled_data.next())
        self.init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
        for m, callbacks in enumerate(per_model_callbacks):
            for callback in callbacks:
                callback.initialize(
                    self, self.models[m], 'model_%d' % m, working_dir,
                    summary_writer)
        for callback in combined_model_callbacks:
            callback.initialize(
                self, self.combined_model, 'combined_model', working_dir,
                summary_writer)
        untrained_models = list(range(len(models)))
        prev_loss = [sys.float_info.max] * len(models)
        iter_below_tol = [0] * len(models)
        step = 0
        while True:
            if step == self.first_consensus:
                self.consensus_method.train(
                    state=self.consensus_state, session=self.session,
                    labeled_data=labeled_data, unlabeled_data=unlabeled_data,
                    max_iter=self.first_consensus_max_iter)
                self.session.run(fetches=self._consensus_loss_weight_var.assign(
                    self.consensus_loss_weight))
            elif step > self.first_consensus \
                    and (step - self.first_consensus) \
                    % self.consensus_update_frequency == 0:
                self.consensus_method.train(
                    state=self.consensus_state, session=self.session,
                    labeled_data=labeled_data, unlabeled_data=unlabeled_data,
                    max_iter=self.consensus_update_max_iter)
            fetches = [[models[m].train_op, models[m].loss_op]
                       for m in untrained_models]
            run_outputs = self.session.run(fetches=fetches, feed_dict=feed_dict)
            losses = []
            for i, m in enumerate(untrained_models[:]):
                loss = run_outputs[i][1]
                for callback in per_model_callbacks[m]:
                    callback(
                        self.session, feed_dict, loss, self.train_iteration[m])
                loss_diff = abs(prev_loss[m] - loss)
                if loss_diff < abs_loss_chg_tol[m] \
                        or abs(loss_diff / prev_loss[m]) < rel_loss_chg_tol[m]:
                    iter_below_tol[m] += 1
                else:
                    iter_below_tol[m] = 0
                if iter_below_tol[m] >= loss_chg_iter_below_tol[m]:
                    logger.info('Model %d finished training: Loss value '
                                'converged.' % m)
                    untrained_models.remove(m)
                elif step >= max_iter[m] - 1:
                    logger.info('Model %d finished training: Maximum number of '
                                'iterations reached.' % m)
                    untrained_models.remove(m)
                prev_loss[m] = loss
                self.train_iteration[m] += 1
                losses.append(loss)
            for callback in combined_model_callbacks:
                callback(
                    self.session, feed_dict, np.mean(losses),
                    max(self.train_iteration))
            if len(untrained_models) == 0:
                logger.info('All models have finished training.')
                break
            feed_dict = _get_feed_dict(
                models=self.models, labeled_data=labeled_data.next(),
                unlabeled_data=unlabeled_data.next())
            step += 1
        self.consensus_method.train(
            state=self.consensus_state, session=self.session,
            labeled_data=labeled_data, unlabeled_data=unlabeled_data,
            max_iter=self.consensus_update_max_iter)
        if save_trained:
            learners.Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=step)

    @property
    @memoize
    def combined_model(self):
        return CombinedModel(
            models=self.models, combined_outputs=self.consensus,
            copy_models=False)
