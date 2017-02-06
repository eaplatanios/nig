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

"""The ops included in this module were borrowed from the TensorFlow repository
and refactored slightly."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

__author__ = 'eaplatanios'

__all__ = ['broadcast_weights']

__ASSERT_BROADCASTABLE_ERROR_PREFIX__ = 'The provided weights can not be ' \
                                        'broadcast to the provided values'


def _has_valid_dims(weights_shape, values_shape, name=None):
    with tf.name_scope(
            name, 'has_invalid_dims', (weights_shape, values_shape)) as scope:
        values_shape_2d = tf.expand_dims(values_shape, -1)
        valid_dims = tf.concat(
            values=(values_shape_2d, tf.ones_like(values_shape_2d)), axis=1)
        weights_shape_2d = tf.expand_dims(weights_shape, -1)
        invalid_dims = tf.sets.set_difference(weights_shape_2d, valid_dims)
        num_invalid_dims = tf.size(invalid_dims.values, name='num_invalid_dims')
        return tf.equal(0, num_invalid_dims, name=scope)


def _has_valid_non_scalar_shape(
        weights_rank, weights_shape, values_rank, values_shape, name=None):
    with tf.name_scope(
            name, 'has_valid_non_scalar_shape',
            (weights_rank, weights_shape, values_rank, values_shape)) as scope:
        is_same_rank = tf.equal(values_rank, weights_rank, name='is_same_rank')
        return tf.cond(
            is_same_rank,
            lambda: _has_valid_dims(weights_shape, values_shape),
            lambda: is_same_rank,
            name=scope)


def _assert_broadcastable(weights, values, name=None):
    """Asserts that `weights` can be broadcast to `values`.

    The weights can be either a scalar or a tensor with the same rank as the
    target values and with each dimension being equal to `1` or to the
    corresponding target values dimension (so that they are broadcastable).

    Args:
      weights (tf.Tensor): Tensor containing the weights.
      values (tf.Tensor): Tensor containing the values.

    Returns:
        tf.Operation: Op raising an `InvalidArgumentError`m if `weights` has
            a non-broadcastable shape, or:

        `tf.no_op`: Op that does nothing, if static checks determine that
            `weights` has a broadcastable shape.

    Raises:
        ValueError: If static checks determine tht `weights` has a
        non-broadcastable shape.
    """
    with tf.name_scope(name, 'assert_broadcastable',
                       (weights, values)) as scope:
        with tf.name_scope(None, 'weights', (weights,)) as weights_scope:
            weights = tf.convert_to_tensor(weights, name=weights_scope)
            weights_shape = tf.shape(weights, name='shape')
            weights_rank = tf.rank(weights, name='rank')
        weights_rank_static = tf.contrib.util.constant_value(weights_rank)
        with tf.name_scope(None, 'values', (values,)) as values_scope:
            values = tf.convert_to_tensor(values, name=values_scope)
            values_shape = tf.shape(values, name='shape')
            values_rank = tf.rank(values, name='rank')
        values_rank_static = tf.contrib.util.constant_value(values_rank)

        if weights_rank_static is not None and values_rank_static is not None:
            # Use static shape if known.
            if weights_rank_static == 0:
                return tf.no_op(name='static_scalar_check_success')
            if weights_rank_static != values_rank_static:
                raise ValueError(
                    '%s (values.rank=%s, weights.rank=%s).' % (
                        __ASSERT_BROADCASTABLE_ERROR_PREFIX__,
                        values_rank_static, weights_rank_static))
            weights_shape_static = tf.contrib.util.constant_value(weights_shape)
            values_shape_static = tf.contrib.util.constant_value(values_shape)
            if weights_shape_static is not None \
                    and values_shape_static is not None:
                # Sanity checking. This should always be true since we checked
                # the rank earlier.
                ndims = len(values_shape_static)
                assert ndims == len(weights_shape_static)

                for i in range(ndims):
                    if weights_shape_static[i] not in (1, values_shape_static[i]):
                        raise ValueError(
                            '%s (Mismatch at dim %s; '
                            'values.shape=%s, weights.shape=%s).' % (
                                __ASSERT_BROADCASTABLE_ERROR_PREFIX__, i,
                                values_shape_static, weights_shape_static))
                return tf.no_op(name='static_dims_check_success')

        # Otherwise use dynamic shape.
        is_scalar = tf.equal(0, weights_rank, name='is_scalar')
        data = (
            __ASSERT_BROADCASTABLE_ERROR_PREFIX__,
            'weights.shape=', weights.name, weights_shape,
            'values.shape=', values.name, values_shape,
            'is_scalar=', is_scalar)
        is_valid_shape = tf.cond(
            is_scalar,
            lambda: is_scalar,
            lambda: _has_valid_non_scalar_shape(
                weights_rank, weights_shape, values_rank, values_shape),
            name='is_valid_shape')
        return tf.Assert(is_valid_shape, data, name=scope)


def broadcast_weights(weights, values, name=None):
    """Broadcasts `weights` to the same shape as `values`.

    This returns a version of `weights` following the same broadcast rules as
    `mul(weights, values)`, but limited to the weights shapes allowed by
    `assert_broadcastable`. When computing a weighted average, use this function
    to broadcast `weights` before summing them; e.g.,
    `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.
    Args:
      weights: `Tensor` whose shape is broadcastable to `values` according to the
        rules of `assert_broadcastable`.
      values: `Tensor` of any shape.
    Returns:
      `weights` broadcast to `values` shape according to the rules of
        `assert_broadcastable`.
    """
    with tf.name_scope(name, 'broadcast_weights', [weights, values]) as scope:
        values = tf.convert_to_tensor(values, name='values')
        weights = tf.convert_to_tensor(
            weights, dtype=values.dtype.base_dtype, name='weights')

        # Try static check for exact match.
        weights_shape = weights.get_shape()
        values_shape = values.get_shape()
        if weights_shape.is_fully_defined() \
                and values_shape.is_fully_defined() \
                and weights_shape.is_compatible_with(values_shape):
            return weights

        with tf.control_dependencies((_assert_broadcastable(weights, values),)):
            return tf.multiply(weights, tf.ones_like(values), name=scope)
