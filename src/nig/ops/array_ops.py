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

__all__ = ['roll_axis']


def roll_axis(tensor, axis, target=0):
    """Rolls the specified axis, until it lies in the provided position. The
    positions of the other axes do not change relative to one another.

    Args:
        tensor (tf.Tensor): Tensor whose axis to roll.
        axis (int): Index of the axis to roll.
        target (int, optional): Optional target axis index. The axis is rolled
            until it lies at this position. Defaults to `0`, resulting in a
            "complete" roll.

    Returns:
        tf.Tensor: Tensor with rolled axes.
    """
    if axis == target:
        return tensor
    rank = tf.rank(tensor)
    if axis < 0:
        axis = rank + axis
    else:
        axis = tf.constant(axis)
    if target < 0:
        target = rank + target
    else:
        target = tf.constant(target)

    def _roll_forward():
        return tf.concat_v2(
            axis=0, values=[tf.range(0, axis),
                            tf.range(axis + 1, target + 1), axis[None],
                            tf.range(target + 1, rank)])

    def _roll_backward():
        return tf.concat_v2(
            axis=0, values=[tf.range(0, target), axis[None],
                            tf.range(target, axis),
                            tf.range(axis + 1, rank)])

    perm = tf.cond(axis < target, _roll_forward, _roll_backward)
    return tf.transpose(tensor, perm=perm)
