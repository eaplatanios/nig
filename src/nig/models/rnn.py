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

__all__ = ['dynamic_hierarchical_rnn']


def dynamic_hierarchical_rnn(cells, periods, inputs, sequence_length=None,
                             input_levels=None, inputs_reduction=None,
                             initial_states=None, dtype=None,
                             parallel_iterations=None, swap_memory=False,
                             time_major=False, scope=None):
    """

    Args:
        cells:
        periods:
        inputs:
        sequence_length:
        input_levels:
        inputs_reduction (function): Needs to be ones of TensorFlow's
            reduction ops (i.e., tf.reduce_*).
        initial_states:
        dtype:
        parallel_iterations:
        swap_memory:
        time_major:
        scope:

    Returns:

    """
    # Check validity of provided cells and periods
    if not isinstance(cells, list):
        raise TypeError('cells must be a list of tf.nn.rnn_cell.RNNCells.')
    if not isinstance(periods, list):
        raise TypeError('periods must be a list of ints.')
    if any(not isinstance(period, int) for period in periods):
        raise TypeError('periods must be a list of ints.')
    if len(cells) != len(periods):
        raise ValueError('The number of cells must match that of periods.')
    if input_levels is not None:
        error_msg = 'input_levels must be a list of ints of lists of ints.'
        if not isinstance(input_levels, list):
            raise TypeError(error_msg)
        for levels in input_levels:
            if levels is not None \
                    and not isinstance(levels, list) \
                    and not isinstance(levels, int):
                raise TypeError(error_msg)
            if not isinstance(levels, int) \
                    and any(not isinstance(input_level, int)
                            for input_level in levels):
                raise TypeError(error_msg)
    else:
        # -1 means the previous level.
        input_levels = [0] + [-1] * (len(cells) - 1)
    input_levels = [l if l is not None or (isinstance(l, list) and len(l) > 0)
                    else 0 for l in input_levels]
    if len(cells) != len(input_levels):
        raise ValueError('The number of cells must match that of level_inputs.')
    if initial_states is not None:
        if len(cells) != len(initial_states):
            raise ValueError('The number of cells must match that of '
                             'initial states.')
    else:
        initial_states = [None] * len(cells)

    # Determine the time axis in the input tensors
    if not time_major:
        time_axis = 1
    else:
        time_axis = 0

    unpacked_inputs = tf.unstack(inputs, axis=time_axis)
    level_outputs = []
    level_states = []
    for level_index, cell, period, in_levels, initial_state \
            in zip(range(len(cells)), cells, periods, input_levels,
                   initial_states):
        # Obtain the inputs for the current level
        def _get_level_output(level):
            if level_index == 0:
                assert level == 0, 'The first level can only depend on the ' \
                                   'input time series.'
            else:
                assert level <= len(level_outputs), 'Levels can only depend ' \
                                                    'on other previous levels.'
            if level == -1:
                level = level_index - 1
            if level != 0:
                period_msg = 'Periods must be positive multiples of the ' \
                             'corresponding input level periods.'
                assert period >= periods[level - 1], period_msg
                assert period % periods[level - 1] == 0, period_msg
                step = period // periods[level - 1]
                outputs = level_outputs[level - 1]
            else:
                step = period
                outputs = unpacked_inputs
            return tf.stack(outputs[::step], axis=time_axis)
        if isinstance(in_levels, list):
            if inputs_reduction is None:
                inputs_ = tf.concat_v2(
                    axis=2, values=[_get_level_output(l) for l in in_levels])
            else:
                inputs_ = tf.stack(
                    [_get_level_output(l) for l in in_levels], axis=0)
                inputs_ = inputs_reduction(inputs_, axis=0)
        else:
            inputs_ = _get_level_output(in_levels)
        sequence_length_ = tf.divide(sequence_length, period)

        # Create RNN for the current level
        output, state = tf.nn.dynamic_rnn(
            cell=cell, inputs=inputs_, sequence_length=sequence_length_,
            initial_state=initial_state, dtype=dtype,
            parallel_iterations=parallel_iterations, swap_memory=swap_memory,
            time_major=time_major, scope=scope + '/level_%d' % level_index)
        level_outputs.append(tf.unstack(output, axis=time_axis))
        level_states.append(state)
    return [tf.stack(o, axis=time_axis) for o in level_outputs], level_states


def rolling_window_rnn(cells, window_length, window_offset, inputs,
                       sequence_length=None, initial_states=None,
                       dtype=None, parallel_iterations=None, swap_memory=False,
                       time_major=False, scope=None):
    # Check validity of provided cells, the window length and offset.
    if not isinstance(cells, list):
        raise TypeError('cells must be a list of tf.nn.rnn_cell.RNNCells.')
    if not isinstance(window_length, int):
        raise TypeError('window length must be an int.')
    if not isinstance(window_offset, int):
        raise TypeError('window offset must be an int.')
    #if window_length > sequence_length:
    #    raise TypeError('window length must be at most sequence length.')
    num_windows = (inputs.get_shape().as_list()[1] - window_length) // \
                  window_offset + 1
    if len(cells) != num_windows:
        raise ValueError('The number of cells must match sequence_length // '
                         '\window_length.')
    if initial_states is not None:
        if len(cells) != len(initial_states):
            raise ValueError('The number of cells must match that of '
                             'initial states.')
    else:
        initial_states = [None] * len(cells)

    # Determine the time axis in the input tensors
    if not time_major:
        time_axis = 1
    else:
        time_axis = 0

    unpacked_inputs = tf.unstack(inputs, axis=time_axis)
    window_outputs = []
    window_states = []
    for window_index in range(num_windows):
        # Obtain the inputs for the current cell
        time_first_index = window_index * window_offset
        time_first_index_next = time_first_index + window_length
        cell_input = tf.stack(
            unpacked_inputs[time_first_index:time_first_index_next],
            axis=time_axis)
        output, state = tf.nn.dynamic_rnn(
            cell=cells[window_index], inputs=cell_input,
            sequence_length=tf.ones_like(sequence_length) *
                            tf.constant(window_length),
            initial_state=initial_states[window_index], dtype=dtype,
            parallel_iterations=parallel_iterations, swap_memory=swap_memory,
            time_major=time_major, scope=scope + '/window_%d' % window_index)
        window_outputs.append(tf.unstack(output, axis=time_axis))
        window_states.append(state)

    return [tf.stack(o, axis=time_axis) for o in window_outputs], window_states