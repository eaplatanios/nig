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

from ..utilities.functions import pipeline

__author__ = 'eaplatanios'


@pipeline(min_num_args=1)
def norm_summary(tensors, name=None):
    if not isinstance(tensors, list):
        tensors = [tensors]
    if name is None:
        name = '_'.join([t.op.name for t in tensors]) + '_norm'
    tensors_norm = tf.reduce_sum(
        [tf.nn.l2_loss(tensor) for tensor in tensors], name=name)
    tf.scalar_summary(tensors_norm._output_op.name, tensors_norm)
    return tensors


@pipeline(min_num_args=2)
def norm_clipping(tensors, clip_norm):
    return [tf.clip_by_norm(t, clip_norm) for t in tensors]


@pipeline(min_num_args=2)
def average_norm_clipping(tensors, clip_norm):
    return [tf.clip_by_average_norm(t, clip_norm) for t in tensors]


@pipeline(min_num_args=2)
def global_norm_clipping(tensors, clip_norm):
    tensors, _ = tf.clip_by_global_norm(tensors, clip_norm)
    return tensors
