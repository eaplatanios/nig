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
    """Rolls the specified axis backwards, until it lies in the provided
    position. The positions of the other axes do not change relative to one
    another.

    Args:
        tensor (tf.Tensor): Tensor whose axis to roll.
        axis (int): Index of the axis to roll backwards.
        target (int, optional): Optional target axis index. The axis is rolled
            until it lies at this position. Defaults to `0`, resulting in a
            "complete" roll.

    Notes:
        This method replicates the functionality of the respective method in
        numpy. More information about that method can be found at:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html

    Returns:
        tf.Tensor: Tensor with rolled axes.
    """
    rank = tf.rank(tensor)
    # ndims = tensor.get_shape().ndims
    # if ndims is None:
    #     raise ValueError('Could not infer the tensor rank.')
    if axis < 0:
        axis = rank + axis
    else:
        axis = tf.constant(axis)
    if target < 0:
        target = rank + target
    else:
        target = tf.constant(target)
    # TODO: Raise informative exception in case of invalid arguments.
    # if not (0 <= axis < rank):
    #     raise ValueError('roll_axis: axis (%d) must be >= 0 and < %d.'
    #                      % (axis, rank))
    # if not (0 <= target < rank + 1):
    #     raise ValueError('roll_axis: start (%d) must be >= 0 and < %d.'
    #                      % (target, rank + 1))
    target = tf.cond(axis < target, lambda: target - 1, lambda: target)

    def _roll_axis():
        perm = [axis[None], tf.range(0, axis), tf.range(axis + 1, rank)]
        perm = tf.concat(concat_dim=0, values=perm)
        return tf.transpose(tensor, perm=perm)

    return tf.cond(tf.equal(axis, target), lambda: tensor, _roll_axis)
