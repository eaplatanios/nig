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
    def __init__(self, inputs, num_hidden, activation=tf.nn.sigmoid,
                 mean_field=True, num_samples=100, cd_steps=1,
                 reconstruction_error_summary=False, optimizer=None,
                 optimizer_opts=None, graph=None):
        self.graph = inputs.graph if graph is None else graph
        num_input = tf.shape(inputs)[1]
        self.inputs = inputs
        self.num_hidden = num_hidden
        self.activation = activation
        self.mean_field = mean_field
        self.num_samples = num_samples
        self.cd_steps = cd_steps
        if optimizer is None:
            optimizer = lambda: tf.train.AdamOptimizer()
        if optimizer_opts is None:
            optimizer_opts = {
                'batch_size': 100, 'max_iter': 1000, 'abs_loss_chg_tol': 1e-6,
                'rel_loss_chg_tol': 1e-3, 'loss_chg_iter_below_tol': 5,
                'grads_processor': None}
        with self.graph.as_default():
            p = tf.reduce_mean(self.inputs, reduction_indices=[0])
            self.vb = tf.Variable(
                initial_value=tf.log(p / (1 - p)), validate_shape=False,
                name='vb')
            self.hb = tf.Variable(tf.zeros(
                shape=[self.num_hidden], dtype=tf.float32),
                validate_shape=False, name='hb')
            self.w = tf.Variable(tf.random_normal(
                shape=[num_input, self.num_hidden], mean=0.0, stddev=0.01,
                dtype=tf.float32), validate_shape=False, name='w')
            h_prediction, reconstruction_error, gradients = self._gradients()
        super(RBM, self).__init__(
            inputs=inputs, outputs=h_prediction, train_outputs=[],
            loss=reconstruction_error,
            loss_summary=reconstruction_error_summary, gradients=gradients,
            optimizer=optimizer, optimizer_opts=optimizer_opts,
            graph=self.graph)

    @graph_context
    def _conditional_h_given_v(self, v):
        return self.activation(tf.add(self.hb, tf.matmul(v, self.w)))

    @graph_context
    def _conditional_v_given_h(self, h):
        return self.activation(
            tf.add(self.vb, tf.matmul(h, self.w, transpose_b=True)))

    @staticmethod
    @graph_context
    def _sample_binary(p):
        uniform = tf.random_uniform(
            shape=tf.shape(p), minval=0, maxval=1, dtype=tf.float32)
        return tf.nn.relu(tf.sign(tf.sub(p, uniform)))

    @graph_context
    def _gradients(self):
        v = self.inputs
        with tf.name_scope('rbm'):
            h_p = self._conditional_h_given_v(v)
            if self.mean_field:
                h_prediction = h_p
                v_prediction = self._conditional_v_given_h(h_prediction)
            else:
                h_samples = []
                v_samples = []
                for sample in range(self.num_samples):
                    h_sample = self._sample_binary(h_p)
                    v_p = self._conditional_v_given_h(h_p)
                    v_sample = self._sample_binary(v_p)
                    h_samples.append(h_sample)
                    v_samples.append(v_sample)
                h_samples = tf.pack(h_samples, axis=0)
                v_samples = tf.pack(v_samples, axis=0)
                h_prediction = tf.reduce_mean(h_samples, reduction_indices=[0])
                v_prediction = tf.reduce_mean(v_samples, reduction_indices=[0])
            reconstruction_error = tf.reduce_mean(
                tf.abs(self.inputs - v_prediction), reduction_indices=[1])
            reconstruction_error = tf.reduce_mean(reconstruction_error)
            h = self._sample_binary(h_p)
            vb_data_grad = tf.reduce_sum(v, reduction_indices=[0])
            hb_data_grad = tf.reduce_sum(h_p, reduction_indices=[0])
            w_data_grad = tf.batch_matmul(v[:, :, None], h[:, None, :])
            w_data_grad = tf.reduce_sum(w_data_grad, reduction_indices=[0])
            for cd_step in range(self.cd_steps):
                v = self._sample_binary(self._conditional_v_given_h(h))
                h = self._sample_binary(self._conditional_h_given_v(v))
            vb_model_grad = tf.reduce_sum(v, reduction_indices=[0])
            hb_model_grad = tf.reduce_sum(h, reduction_indices=[0])
            w_model_grad = tf.batch_matmul(v[:, :, None], h[:, None, :])
            w_model_grad = tf.reduce_sum(w_model_grad, reduction_indices=[0])
            vb_grad = vb_model_grad - vb_data_grad
            hb_grad = hb_model_grad - hb_data_grad
            w_grad = w_model_grad - w_data_grad
            gradients = {self.vb: vb_grad, self.hb: hb_grad, self.w: w_grad}
            return h_prediction, reconstruction_error, gradients
