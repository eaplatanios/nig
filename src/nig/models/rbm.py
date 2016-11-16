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

import tensorflow as tf

from ..learning.models import Model
from ..utilities.tensorflow import graph_context

__author__ = 'eaplatanios'

__all__ = ['RBM']


class RBM(Model):
    def __init__(self, inputs, num_hidden, mean_field=True, num_samples=100,
                 mean_field_cd=False, cd_steps=1, loss_summary=False,
                 optimizer=None, optimizer_opts=None, graph=None):
        self.graph = inputs.graph if graph is None else graph
        self.num_inputs = inputs.get_shape()[1].value
        self.inputs = inputs
        self.num_hidden = num_hidden
        self.mean_field = mean_field
        self.num_samples = num_samples
        self.mean_field_cd = mean_field_cd
        self.cd_steps = cd_steps
        if optimizer is None:
            optimizer = lambda: tf.train.AdamOptimizer()
        if optimizer_opts is None:
            optimizer_opts = {
                'batch_size': 100, 'max_iter': 1000, 'abs_loss_chg_tol': 1e-6,
                'rel_loss_chg_tol': 1e-3, 'loss_chg_iter_below_tol': 5,
                'grads_processor': None}
        with self.graph.as_default():
            # p = tf.reduce_mean(self.inputs, reduction_indices=[0])
            # self.vb = tf.Variable(
            #     initial_value=tf.log(p / (1 - p)), name='vb')
            self.vb = tf.Variable(tf.zeros(
                shape=[self.num_inputs], dtype=tf.float32), name='vb')
            self.hb = tf.Variable(tf.zeros(
                shape=[self.num_hidden], dtype=tf.float32), name='hb')
            self.w = tf.Variable(tf.random_normal(
                shape=[self.num_inputs, self.num_hidden], mean=0.0, stddev=0.01,
                dtype=tf.float32), name='w')
        outputs = self._outputs()
        loss = self._loss()
        super(RBM, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=[],
            loss=loss, loss_summary=loss_summary,
            optimizer=optimizer, optimizer_opts=optimizer_opts,
            graph=self.graph)

    @graph_context
    def _conditional_h_given_v(self, v):
        return tf.nn.sigmoid(tf.add(self.hb, tf.matmul(v, self.w)))

    @graph_context
    def _conditional_v_given_h(self, h):
        return tf.nn.sigmoid(
            tf.add(self.vb, tf.matmul(h, self.w, transpose_b=True)))

    @staticmethod
    @graph_context
    def _sample_binary(p):
        return tf.nn.relu(tf.sign(p - tf.random_uniform(tf.shape(p), 0, 1)))

    @graph_context
    def _contrastive_divergence(self, k=1, initial_v=None):
        """Runs a `k`-step Gibbs sampling chain to sample from the probability
        distribution of the RBM.

        Args:
            k (int, optional): Optional number of Gibbs sampling steps. Defaults
                to `1`.
            initial_v (tf.Tensor, optional): Optional initial visible units
                samples. If set to `None`, it is assigned to `self.inputs`.
                Defaults to `None`.

        Returns:
            tf.Tensor: Visible state sample tensor.
        """
        v = self.inputs if initial_v is None else initial_v
        for _ in range(k):
            h_p = self._conditional_h_given_v(v)
            h = h_p if self.mean_field_cd else self._sample_binary(h_p)
            v_p = self._conditional_v_given_h(h)
            v = v_p if self.mean_field_cd else self._sample_binary(v_p)
        v = tf.stop_gradient(v)
        return v

    @graph_context
    def _free_energy(self, v):
        cond_term = tf.log(1 + tf.exp(tf.add(self.hb, tf.matmul(v, self.w))))
        cond_term = -tf.reduce_sum(
            cond_term, reduction_indices=[1], keep_dims=True)
        bias_term = -tf.matmul(v, tf.transpose(self.vb[None]))
        return tf.add(cond_term, bias_term)

    @graph_context
    def _loss(self):
        v_sample = self._contrastive_divergence(self.cd_steps)
        v_free_energy = self._free_energy(self.inputs)
        v_sample_free_energy = self._free_energy(v_sample)
        return tf.reduce_mean(tf.sub(v_free_energy, v_sample_free_energy))

    @graph_context
    def _outputs(self):
        with tf.name_scope('rbm'):
            h_p = self._conditional_h_given_v(self.inputs)
            if self.mean_field:
                return h_p
            else:
                h_samples = []
                for sample in range(self.num_samples):
                    h_sample = self._sample_binary(h_p)
                    v_p = self._conditional_v_given_h(h_p)
                    v_sample = self._sample_binary(v_p)
                    h_p = self._conditional_h_given_v(v_sample)
                    h_samples.append(h_sample)
                h_samples = tf.pack(h_samples, axis=0)
                return tf.reduce_mean(h_samples, reduction_indices=[0])


class SemiSupervisedRBM(Model):
    def __init__(self, inputs, num_hidden, mean_field=True, num_samples=100,
                 mean_field_cd=False, cd_steps=1, loss_summary=False,
                 optimizer=None, optimizer_opts=None, graph=None):
        self.graph = inputs.graph if graph is None else graph
        self.num_inputs = inputs.get_shape()[1].value
        self.inputs = inputs
        self.num_hidden = num_hidden
        self.mean_field = mean_field
        self.num_samples = num_samples
        self.mean_field_cd = mean_field_cd
        self.cd_steps = cd_steps
        self.train_outputs = tf.placeholder_with_default(
            input=tf.zeros(shape=(0,), dtype=tf.float32), shape=[None])
        if optimizer is None:
            optimizer = lambda: tf.train.AdamOptimizer()
        if optimizer_opts is None:
            optimizer_opts = {
                'batch_size': 100, 'max_iter': 1000, 'abs_loss_chg_tol': 1e-6,
                'rel_loss_chg_tol': 1e-3, 'loss_chg_iter_below_tol': 5,
                'grads_processor': None}
        with self.graph.as_default():
            self.vb = tf.Variable(tf.zeros(
                shape=[self.num_inputs], dtype=tf.float32), name='vb')
            self.hb = tf.Variable(tf.zeros(
                shape=[self.num_hidden], dtype=tf.float32), name='hb')
            self.w = tf.Variable(tf.random_normal(
                shape=[self.num_inputs, self.num_hidden], mean=0.0, stddev=0.01,
                dtype=tf.float32), name='w')
        outputs = self._outputs()
        loss = self._loss()
        super(SemiSupervisedRBM, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=self.train_outputs,
            loss=loss, loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts, graph=self.graph)

    @graph_context
    def _conditional_h_given_v(self, v):
        return tf.nn.sigmoid(tf.add(self.hb, tf.matmul(v, self.w)))

    @graph_context
    def _conditional_v_given_h(self, h):
        return tf.nn.sigmoid(
            tf.add(self.vb, tf.matmul(h, self.w, transpose_b=True)))

    @staticmethod
    @graph_context
    def _sample_binary(p):
        return tf.nn.relu(tf.sign(p - tf.random_uniform(tf.shape(p), 0, 1)))

    @graph_context
    def _contrastive_divergence(self, k=1, initial_v=None):
        """Runs a `k`-step Gibbs sampling chain to sample from the probability
        distribution of the RBM.

        Args:
            k (int, optional): Optional number of Gibbs sampling steps. Defaults
                to `1`.
            initial_v (tf.Tensor, optional): Optional initial visible units
                samples. If set to `None`, it is assigned to `self.inputs`.
                Defaults to `None`.

        Returns:
            tf.Tensor: Visible state sample tensor.
        """
        v = self.inputs if initial_v is None else initial_v
        h_p = self._conditional_h_given_v(v)
        h = h_p if self.mean_field_cd else self._sample_binary(h_p)
        for _ in range(k):
            v_p = self._conditional_v_given_h(h)
            v = v_p if self.mean_field_cd else self._sample_binary(v_p)
            h_p = self._conditional_h_given_v(v)
            h = h_p if self.mean_field_cd else self._sample_binary(h_p)
        v = tf.stop_gradient(v)
        h = tf.stop_gradient(h)
        return v, h

    @graph_context
    def _free_energy(self, v, h=None):
        if h is None:
            h_given_v_term = tf.add(self.hb, tf.matmul(v, self.w))
            cond_term = tf.log(1 + tf.exp(h_given_v_term))
            cond_term = tf.reduce_sum(
                cond_term, reduction_indices=[1], keep_dims=True)
            bias_term = tf.matmul(v, self.vb[:, None])
            return -tf.add(cond_term, bias_term)
        v_bias_term = tf.matmul(v, self.vb[:, None])
        h_bias_term = tf.matmul(h, self.hb[:, None])
        w_term = tf.batch_matmul(
            h[:, None, :], tf.matmul(v, self.w)[:, :, None])[:, :, 0]
        return -tf.add_n((v_bias_term, h_bias_term, w_term))

    @graph_context
    def _unlabeled_loss(self, unlabeled_inputs=None):
        if unlabeled_inputs is None:
            unlabeled_inputs = self.inputs
        v_sample, _ = self._contrastive_divergence(
            k=self.cd_steps, initial_v=unlabeled_inputs)
        data_free_energy = self._free_energy(unlabeled_inputs)
        sample_free_energy = self._free_energy(v_sample)
        return tf.reduce_sum(tf.sub(data_free_energy, sample_free_energy))

    @graph_context
    def _labeled_loss(self, labeled_inputs, labels):
        v_sample, h_sample = self._contrastive_divergence(
            k=self.cd_steps, initial_v=labeled_inputs)
        data_free_energy = self._free_energy(labeled_inputs, labels)
        sample_free_energy = self._free_energy(v_sample, h_sample)
        return tf.reduce_sum(tf.sub(data_free_energy, sample_free_energy))

    @graph_context
    def _combined_loss(self):
        unlabeled_mask = self.train_outputs < 0
        labeled_mask = tf.logical_not(unlabeled_mask)
        unlabeled_inputs = tf.boolean_mask(self.inputs, unlabeled_mask)
        unlabeled_loss = self._unlabeled_loss(unlabeled_inputs)
        labeled_inputs = tf.boolean_mask(self.inputs, labeled_mask)
        labels = tf.boolean_mask(self.train_outputs, labeled_mask)
        labeled_loss = self._labeled_loss(labeled_inputs, labels[:, None])
        return tf.add(unlabeled_loss, labeled_loss)

    @graph_context
    def _loss(self):
        num_train_outputs = tf.shape(self.train_outputs)[0]
        return tf.cond(
            num_train_outputs > 0, lambda: self._combined_loss(),
            lambda: self._unlabeled_loss())

    @graph_context
    def _outputs(self):
        with tf.name_scope('rbm'):
            h_p = self._conditional_h_given_v(self.inputs)
            if self.mean_field:
                return h_p
            else:
                h_samples = []
                for sample in range(self.num_samples):
                    h_sample = self._sample_binary(h_p)
                    v_p = self._conditional_v_given_h(h_p)
                    v_sample = self._sample_binary(v_p)
                    h_p = self._conditional_h_given_v(v_sample)
                    h_samples.append(h_sample)
                h_samples = tf.pack(h_samples, axis=0)
                return tf.reduce_mean(h_samples, reduction_indices=[0])
