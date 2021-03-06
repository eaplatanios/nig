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

from .complex import mod_relu, matmul_parameterized_unitary, zero_complex

__author__ = 'eaplatanios'

__all__ = ['URNNCell']


class URNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, state_is_tuple=True):
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        c = zero_complex(shape=[batch_size, self._num_units], dtype=dtype)
        if self._state_is_tuple:
            h = tf.zeros(shape=[batch_size, self._num_units], dtype=dtype)
            return c, h
        return c

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                state = state[0]
            w_in = tf.get_variable(
                name='W_in', shape=[inputs.get_shape()[1], self._num_units * 2])
            in_proj = tf.matmul(inputs, w_in)
            in_proj_c = tf.complex(
                in_proj[:, :self._num_units], in_proj[:, self._num_units:])
            next_state = mod_relu(
                in_proj_c + matmul_parameterized_unitary(state),
                tf.get_variable(
                    name='B', dtype=tf.float32, shape=[self._num_units],
                    initializer=tf.constant_initializer(0.0)))
            out = tf.concat(
                values=[tf.real(next_state), tf.imag(next_state)], axis=1)
            w_out = tf.get_variable(
                name='W_out', shape=[self._num_units * 2, self.output_size])
            b_out = tf.get_variable(
                name='B_out', dtype=tf.float32, shape=[self.output_size],
                initializer=tf.constant_initializer(0.0))
            out = tf.matmul(out, w_out) + b_out
        if self._state_is_tuple:
            return out, (next_state, out)
        return out, next_state


# def _linear(args, output_size, bias, bias_start=0.0, scope=None):
#     """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
#
#     Args:
#       args: a 2D Tensor or a list of 2D, batch x n, Tensors.
#       output_size: int, second dimension of W[i].
#       bias: boolean, whether to add a bias term or not.
#       bias_start: starting value to initialize the bias; 0 by default.
#       scope: VariableScope for the created subgraph; defaults to "Linear".
#
#     Returns:
#       A 2D Tensor with shape [batch x output_size] equal to
#       sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
#
#     Raises:
#       ValueError: if some of the arguments has unspecified or wrong shape.
#     """
#     if args is None or (tf.nest.is_sequence(args) and not args):
#         raise ValueError('args must be specified.')
#     if not tf.nest.is_sequence(args):
#         args = [args]
#
#     # Calculate the total size of arguments on dimension 1.
#     total_arg_size = 0
#     shapes = [a.get_shape().as_list() for a in args]
#     for shape in shapes:
#         if len(shape) != 2:
#             raise ValueError('Linear expects 2D arguments: %s' % str(shapes))
#         if not shape[1]:
#             raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
#         else:
#             total_arg_size += shape[1]
#     dtype = [a.dtype for a in args][0]
#
#     # Actual computation
#     with tf.variable_scope(scope or 'linear'):
#         matrix = tf.get_variable(
#             name='Matrix', shape=[total_arg_size, output_size], dtype=dtype)
#         if len(args) == 1:
#             res = tf.matmul(args[0], matrix)
#         else:
#             res = tf.matmul(tf.concat(values=args, axis=1), matrix)
#         if not bias:
#             return res
#         bias_term = tf.get_variable(
#             name='Bias', shape=[output_size], dtype=dtype,
#             initializer=tf.constant_initializer(bias_start, dtype=dtype))
#     return res + bias_term


# class GRUCell(tf.nn.rnn_cell.RNNCell):
#     """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
#     def __init__(self, num_units, activation=tf.tanh, state_is_tuple=True):
#         self._num_units = num_units
#         self._activation = activation
#         self._state_is_tuple = state_is_tuple
#
#     @property
#     def state_size(self):
#         return self._num_units
#
#     @property
#     def output_size(self):
#         return self._num_units
#
#     def __call__(self, inputs, state, scope=None):
#         """Gated recurrent unit (GRU) with num_units cells."""
#         with tf.variable_scope(scope or type(self).__name__):
#             with tf.variable_scope('Gates'):
#                 # We start with bias of 1.0 to not reset and not update.
#                 r, u = tf.split(1, 2, _linear(
#                     [inputs, state], 2 * self._num_units, True, 1.0))
#                 r, u = tf.sigmoid(r), tf.sigmoid(u)
#             with tf.variable_scope("Candidate"):
#                 c = self._activation(_linear(
#                     [inputs, r * state], self._num_units, True))
#             new_h = u * state + (1 - u) * c
#         return new_h, new_h
