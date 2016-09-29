import tensorflow as tf


def dynamic_hierarchical_rnn(cells, periods, inputs, sequence_length=None,
                             input_levels=None, initial_states=None, dtype=None,
                             parallel_iterations=None, swap_memory=False,
                             time_major=False, scope=None):
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

    unpacked_inputs = tf.unpack(inputs, axis=time_axis)
    level_outputs = []
    level_states = []
    for level_index, cell, period, in_levels, initial_state \
            in zip(range(len(cells)), cells, periods, input_levels,
                   initial_states):
        # Obtain the inputs for the current level
        def _get_level_output(level):
            if level_index == 0:
                assert level == 0
            else:
                assert level <= len(level_outputs), 'Levels can only depend ' \
                                                    'on other previous levels.'
            if level == -1:
                level = level_index - 1
            if level != 0:
                assert period >= periods[level - 1]
                assert period % periods[level - 1] == 0
                step = period // periods[level - 1]
                outputs = level_outputs[level - 1]
            else:
                step = period
                outputs = unpacked_inputs
            return tf.pack(outputs[::step], axis=time_axis)
        if isinstance(in_levels, list):
            inputs_ = tf.concat(  # TODO: Support other pooling operations.
                concat_dim=2,
                values=[_get_level_output(l) for l in in_levels])
        else:
            inputs_ = _get_level_output(in_levels)
        sequence_length_ = tf.div(sequence_length, period)

        # Create RNN for the current level
        output, state = tf.nn.dynamic_rnn(
            cell=cell, inputs=inputs_, sequence_length=sequence_length_,
            dtype=dtype, parallel_iterations=parallel_iterations,
            swap_memory=swap_memory, time_major=time_major,
            scope=scope + '/level_%d' % level_index)
        level_outputs.append(tf.unpack(output, axis=time_axis))
        level_states.append(state)
    return [tf.pack(o, axis=time_axis) for o in level_outputs], level_states
