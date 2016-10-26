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

__author__ = 'eaplatanios'

__all__ = ['gradient_descent', 'adam']


def gradient_descent(learning_rate, decay_steps=100, decay_rate=1.0,
                     staircase=False, learning_rate_summary=False,
                     use_locking=False, name='GradientDescent'):
    if decay_rate != 1.0:
        learning_rate_variable = tf.train.exponential_decay(
            learning_rate=float(learning_rate),
            global_step=tf.contrib.framework.get_or_create_global_step(),
            decay_steps=int(decay_steps),
            decay_rate=decay_rate,
            staircase=staircase)
    else:
        learning_rate_variable = learning_rate
    if learning_rate_summary:
        tf.scalar_summary(
            tf.get_default_graph().unique_name(
                name='train/learning_rate', mark_as_used=False),
            learning_rate_variable)
    return tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_variable,
        use_locking=use_locking,
        name=name)


def adam(learning_rate, decay_steps=100, decay_rate=1.0, staircase=False,
         learning_rate_summary=False, beta1=0.9, beta2=0.999, epsilon=1e-8,
         use_locking=False, name='ADAM'):
    if decay_rate != 1.0:
        learning_rate_variable = tf.train.exponential_decay(
            learning_rate=float(learning_rate),
            global_step=tf.contrib.framework.get_or_create_global_step(),
            decay_steps=int(decay_steps),
            decay_rate=decay_rate,
            staircase=staircase)
    else:
        learning_rate_variable = learning_rate
    if learning_rate_summary:
        tf.scalar_summary(
            tf.get_default_graph().unique_name(
                name='train/learning_rate', mark_as_used=False),
            learning_rate_variable)
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate_variable,
        beta1=beta1, beta2=beta2, epsilon=epsilon,
        use_locking=use_locking,
        name=name)
