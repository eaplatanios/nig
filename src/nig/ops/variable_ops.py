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

__all__ = ['create_local']


def create_local(shape, name, trainable=False, collections=None,
                 validate_shape=True, dtype=tf.float32):
    """Creates a new local variable.

    Args:
        shape (tuple): Shape of the variable.
        name (str): Name of the new variable.
        trainable (bool, optional): Optional boolean value indicating whether
            the new variable is trainable or not. Defaults to `False`.
        collections (list(str), optional): Optional list of collection names to
            which the new variable will be added. Defaults to `None`.
        validate_shape (bool, optional): Optional boolean value indicating
            whether to validate the shape of the new variable. Defaults to
            `True`.
        dtype (tf.DType, optional): Optional data type of the new variable.
            Defaults to `tf.float32`.

    Returns:
        tf.Variable: The created variable.
    """
    # Make sure local variables are added to the tf.GraphKeys.LOCAL_VARIABLES
    # collection.
    collections = list(collections or [])
    collections += [tf.GraphKeys.LOCAL_VARIABLES]
    return tf.Variable(
        initial_value=tf.zeros(shape, dtype=dtype), name=name,
        trainable=trainable, collections=collections,
        validate_shape=validate_shape)
