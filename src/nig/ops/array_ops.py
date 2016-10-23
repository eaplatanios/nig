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


def roll_axis(tensor, axis, target=0):
    """Rolls the specified axis backwards, until it lies in the provided
    position. The positions of the other axes do not change relative to one
    another.

    Args:
        tensor (tf.Tensor): Tensor whose axis to roll.
        axis (int): Index of the axis to roll backwards.
        target (int, optional): Optional target axis index. The axis is rolled
            until it lies before this position. Defaults to `0`, resulting in a
            "complete" roll.

    Notes:
        This method replicates the functionality of the respective method in
        numpy. More information about that method can be found at:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html

    Returns:
        tf.Tensor: Tensor with rolled axes.

    Raises:
        ValueError: If the provided axis index or the provided target index is
            outside the boundaries of the tensor dimensionality.
    """
    ndims = tensor.get_shape().ndims
    if ndims is None:
        raise ValueError('Could not infer the tensor rank.')
    if axis < 0:
        axis += ndims
    if target < 0:
        target += ndims
    if not (0 <= axis < ndims):
        raise ValueError('roll_axis: axis (%d) must be >= 0 and < %d.'
                         % (axis, ndims))
    if not (0 <= target < ndims + 1):
        raise ValueError('roll_axis: start (%d) must be >= 0 and < %d.'
                         % (target, ndims + 1))
    if axis < target:
        target -= 1
    if axis == target:
        return tensor
    perm = list(range(0, ndims))
    perm.remove(axis)
    perm.insert(target, axis)
    return tf.transpose(tensor, perm=perm)
