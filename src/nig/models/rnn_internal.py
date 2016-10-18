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


def _infer_state_dtype(explicit_dtype, state):
    if explicit_dtype is not None:
        return explicit_dtype
    elif tf.nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in tf.nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError('Unable to infer dtype from empty state.')
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                'State has tensors of different inferred dtypes. Unable to '
                'infer a single representative dtype.')
        return inferred_dtypes[0]
    return state.dtype


def dynamic_raw_rnn(cell, inputs, sequence_length=None, initial_state=None,
                    dtype=None, parallel_iterations=None, swap_memory=False,
                    time_major=False, scope=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
        raise TypeError('cell must be an instance of tf.nn.rnn_cell.RNNCell.')
    flat_input = tf.nest.flatten(inputs)
    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError('sequence_length must be a vector of length '
                             'batch_size, but encountered shape %s.'
                             % sequence_length.get_shape())

    # By default, the inputs are batch-major shaped: [batch, time, depth].
    # For internal calculations, we switch to time-major: [time, batch, depth].
    if not time_major:
        flat_input = tuple(tf.transpose(fi, [1, 0, 2]) for fi in flat_input)

    # Create a new scope in which the caching device is either determined by
    # the parent scope, or is set to place the cached Variable using the same
    # placement as for the rest of the RNN.
    with tf.variable_scope(scope or 'RNN') as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        # Shape validation for inputs
        input_shape = tuple(tf.shape(input_) for input_ in flat_input)
        time_steps = input_shape[0][0]
        batch_size = input_shape[0][1]
        input_depth = input_shape[0][2]
        for input_ in input_shape:
            if input_[0].get_shape() != time_steps.get_shape():
                raise ValueError('All inputs should have the same time steps.')
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError('All inputs should have the same batch size.')
            if input_[2].get_shape() != batch_size.get_shape():
                raise ValueError('All inputs should have the same depth.')

        # State initialization
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    'If no initial_state is provided, dtype must be.')
            state = cell.zero_state(batch_size, dtype)

        # Shape validation for sequence_length
        def _assert_has_shape(tensor, shape):
            tensor_shape = tf.shape(tensor)
            packed_shape = tf.pack(shape)
            return tf.Assert(
                tf.reduce_all(tf.equal(tensor_shape, packed_shape)),
                ['Expected shape for Tensor ', tensor.name, ' is ',
                 packed_shape, ' but provided shape is ', tensor_shape, '.'])
        if sequence_length is not None:
            with tf.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = tf.identity(
                        sequence_length, name='CheckSeqLen')

    inputs_ta = tf.TensorArray(dtype=inputs.dtype, size=time_steps)
    inputs_ta = inputs_ta.unpack(inputs)

    def _loop_function(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:
            next_cell_state = state
        else:
            next_cell_state = cell_state
        finished = time >= sequence_length
        finished = tf.reduce_all(finished)
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=inputs.dtype),
            lambda: inputs_ta.read(time))
        next_loop_state = None
        return finished, next_input, next_cell_state, emit_output, \
            next_loop_state

    outputs_ta, final_state, _ = raw_rnn(
        cell=cell, loop_function=_loop_function,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        scope=scope)
    outputs = outputs_ta.unpack()

    # Outputs of _dynamic_rnn_loop are always time-major shaped:
    # [time, batch, depth]. If we are performing batch-major calculations,
    # we switch back to batch-major shape: [batch, time, depth].
    if not time_major:
        flat_output = tf.nest.flatten(outputs)
        flat_output = [tf.transpose(output, [1, 0, 2])
                       for output in flat_output]
        outputs = tf.nest.pack_sequence_as(
            structure=outputs, flat_sequence=flat_output)

    return outputs, final_state


def dynamic_hierarchical_rnn(cells, periods, inputs, sequence_length=None,
                             initial_states=None, dtype=None,
                             parallel_iterations=None, swap_memory=False,
                             time_major=False, scope=None):
    # Check validity of provided cells and periods
    if not isinstance(cells, list):
        raise TypeError('cells must be a list of tf.nn.rnn_cell.RNNCells.')
    for cell in cells:
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError('cells must be a list of tf.nn.rnn_cell.RNNCells.')
    if not isinstance(periods, list):
        raise TypeError('periods must be a list of ints.')
    for period in periods:
        if not isinstance(period, int):
            raise TypeError('periods must be a list of ints.')
    if len(cells) != len(periods):
        raise ValueError('The number of cells must match that of periods.')

    # Check validity of provided sequence length
    flat_input = tf.nest.flatten(inputs)
    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError('sequence_length must be a vector of length '
                             'batch_size, but encountered shape %s.'
                             % sequence_length.get_shape())

    # By default, the inputs are batch-major shaped: [batch, time, depth].
    # For internal calculations, we switch to time-major: [time, batch, depth].
    if not time_major:
        flat_input = tuple(tf.transpose(fi, [1, 0, 2]) for fi in flat_input)

    # Create a new scope in which the caching device is either determined by
    # the parent scope, or is set to place the cached Variable using the same
    # placement as for the rest of the RNN.
    with tf.variable_scope(scope or 'RNN') as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        # Shape validation for inputs
        input_shape = tuple(tf.shape(input_) for input_ in flat_input)
        time_steps = input_shape[0][0]
        batch_size = input_shape[0][1]
        input_depth = input_shape[0][2]
        for input_ in input_shape:
            if input_[0].get_shape() != time_steps.get_shape():
                raise ValueError('All inputs should have the same time steps.')
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError('All inputs should have the same batch size.')
            if input_[2].get_shape() != batch_size.get_shape():
                raise ValueError('All inputs should have the same depth.')

        # State initialization
        if initial_states is not None:
            if len(cells) != len(initial_states):
                raise ValueError('The number of cells must match that of '
                                 'initial states.')
            states = initial_states
        else:
            if not dtype:
                raise ValueError(
                    'If no initial_states are provided, dtype must be.')
            states = [cell.zero_state(batch_size, dtype) for cell in cells]

        # Shape validation for sequence_length
        def _assert_has_shape(tensor, shape):
            tensor_shape = tf.shape(tensor)
            packed_shape = tf.pack(shape)
            return tf.Assert(
                tf.reduce_all(tf.equal(tensor_shape, packed_shape)),
                ['Expected shape for Tensor ', tensor.name, ' is ',
                 packed_shape, ' but provided shape is ', tensor_shape, '.'])
        if sequence_length is not None:
            with tf.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = tf.identity(
                    sequence_length, name='CheckSeqLen')

    # TODO: Fill this up from here on.

    inputs_ta = tf.TensorArray(dtype=inputs.dtype, size=time_steps)
    inputs_ta = inputs_ta.unpack(inputs)

    def _loop_function(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:
            next_cell_state = states
        else:
            next_cell_state = cell_state
        finished = time >= sequence_length
        finished = tf.reduce_all(finished)
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=inputs.dtype),
            lambda: inputs_ta.read(time))
        next_loop_state = None
        return finished, next_input, next_cell_state, emit_output, \
               next_loop_state

    outputs_ta, final_state, _ = raw_rnn(
        cell=cell, loop_function=_loop_function,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        scope=scope)
    outputs = outputs_ta.unpack()

    # Outputs of _dynamic_rnn_loop are always time-major shaped:
    # [time, batch, depth]. If we are performing batch-major calculations,
    # we switch back to batch-major shape: [batch, time, depth].
    if not time_major:
        flat_output = tf.nest.flatten(outputs)
        flat_output = [tf.transpose(output, [1, 0, 2])
                       for output in flat_output]
        outputs = tf.nest.pack_sequence_as(
            structure=outputs, flat_sequence=flat_output)

    return outputs, final_state


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
        raise TypeError('cell must be an instance of tf.nn.rnn_cell.RNNCell.')
    flat_input = tf.nest.flatten(inputs)

    # By default, the inputs are batch-major shaped: [batch, time, depth].
    # For internal calculations, we switch to time-major: [time, batch, depth].
    if not time_major:
        flat_input = tuple(tf.transpose(fi, [1, 0, 2]) for fi in flat_input)
    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError('sequence_length must be a vector of length '
                             'batch_size, but encountered shape %s.'
                             % sequence_length.get_shape())
        sequence_length = tf.identity(sequence_length, name='sequence_length')

    # Create a new scope in which the caching device is either determined by
    # the parent scope, or is set to place the cached Variable using the same
    # placement as for the rest of the RNN.
    with tf.variable_scope(scope or 'RNN') as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        # Shape validation for inputs
        input_shape = tuple(tf.shape(input_) for input_ in flat_input)
        batch_size = input_shape[0][1]
        for input_ in input_shape:
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError('All inputs should have the same batch size.')

        # State initialization
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    'If no initial_state is provided, dtype must be.')
            state = cell.zero_state(batch_size, dtype)

        # Shape validation for sequence_length
        def _assert_has_shape(tensor, shape):
            tensor_shape = tf.shape(tensor)
            packed_shape = tf.pack(shape)
            return tf.Assert(
                tf.reduce_all(tf.equal(tensor_shape, packed_shape)),
                ['Expected shape for Tensor ', tensor.name, ' is ',
                 packed_shape, ' but provided shape is ', tensor_shape, '.'])
        if sequence_length is not None:
            with tf.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = tf.identity(
                    sequence_length, name='CheckSeqLen')

        # Dynamic RNN loop
        inputs = tf.nest.pack_sequence_as(
            structure=inputs, flat_sequence=flat_input)
        (outputs, final_state) = _dynamic_rnn_loop(
            cell=cell, inputs=inputs, initial_state=state,
            parallel_iterations=parallel_iterations, swap_memory=swap_memory,
            sequence_length=sequence_length, dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always time-major shaped:
        # [time, batch, depth]. If we are performing batch-major calculations,
        # we switch back to batch-major shape: [batch, time, depth].
        if not time_major:
            flat_output = tf.nest.flatten(outputs)
            flat_output = [tf.transpose(output, [1, 0, 2])
                           for output in flat_output]
            outputs = tf.nest.pack_sequence_as(
                structure=outputs, flat_sequence=flat_output)
        return outputs, final_state


def _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations,
                      swap_memory, sequence_length=None, dtype=None):
    state = initial_state
    if not isinstance(parallel_iterations, int) or parallel_iterations <= 0:
        raise TypeError('parallel_iterations must be a positive integer.')
    state_size = cell.state_size
    flat_input = tf.nest.flatten(inputs)
    flat_output_size = tf.nest.flatten(cell.output_size)

    # Shape validation for inputs
    input_shape = tf.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]
    input_shapes = tuple(fi.get_shape().with_rank_at_least(3)
                         for fi in flat_input)
    const_time_steps, const_batch_size = input_shapes[0].as_list()[:2]
    for shape in input_shapes:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                'Input size (depth of inputs) must be accessible via shape '
                'inference, but encountered None value.')
        current_time_steps = shape[0]
        current_batch_size = shape[1]
        if const_time_steps != current_time_steps:
            raise ValueError(
                'The number of time steps are not the same for all elements in '
                'the inputs of each batch.')
        if const_batch_size != current_batch_size:
            raise ValueError(
                'Batch size is not the same for all elements in the inputs.')

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = tf.nn.rnn_cell._state_size_with_prefix(size, prefix=[batch_size])
        return tf.zeros(tf.pack(size), _infer_state_dtype(dtype, state))
    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = tf.nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = tf.reduce_min(sequence_length)
        max_sequence_length = tf.reduce_max(sequence_length)
    time = tf.constant(0, dtype=tf.int32, name='time')

    # Create input and output tensor arrays
    with tf.name_scope('dynamic_rnn') as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tf.TensorArray(
            dtype=dtype, size=time_steps, tensor_array_name=base_name + name)
    input_ta = tuple(_create_ta('input_%d' % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))
    input_ta = tuple(ta.unpack(fi) for ta, fi in zip(input_ta, flat_input))
    output_ta = tuple(_create_ta('output_%d' % i,
                                 _infer_state_dtype(dtype, state))
                      for i in range(len(flat_output_size)))

    def _time_step(time, output_ta_t, state):
        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, input_shapes):
            input_.set_shape(shape[1:])
        input_t = tf.nest.pack_sequence_as(
            structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, state)
        if sequence_length is not None:
            (output, new_state) = _dynamic_rnn_step(
                time=time, sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output, state=state, call_cell=call_cell,
                state_size=state_size, skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        output = tf.nest.flatten(output)
        output_ta_t = tuple(ta.write(time, out)
                            for ta, out in zip(output_ta_t, output))
        return time + 1, output_ta_t, new_state

    _, output_final_ta, final_state = tf.while_loop(
        cond=lambda time, *_: time < time_steps, body=_time_step,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations, swap_memory=swap_memory)
    final_outputs = tuple(ta.pack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        output.set_shape(tf.nn.rnn_cell._state_size_with_prefix(
            output_size, prefix=[const_time_steps, const_batch_size]))
    final_outputs = tf.nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)
    return final_outputs, final_state


def _dynamic_rnn_step(time, sequence_length, min_sequence_length,
                      max_sequence_length, zero_output, state, call_cell,
                      state_size, skip_conditionals=False):
    # Convert state to a list for ease of use
    flat_state = tf.nest.flatten(state)
    flat_zero_output = tf.nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        return tf.select(time >= sequence_length, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get the
        # previous state and zero output, and which values should get a
        # calculated state and output.
        flat_new_output = [_copy_one_through(zero_output, new_output)
                           for zero_output, new_output
                           in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [_copy_one_through(state, new_state)
                          for state, new_state
                          in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        new_output, new_state = call_cell()
        tf.nest.assert_same_structure(state, new_state)
        flat_new_state = tf.nest.flatten(new_state)
        flat_new_output = tf.nest.flatten(new_output)
        return tf.cond(
            time < min_sequence_length,
            lambda: flat_new_output + flat_new_state,
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps. This is faster when max_seq_len is equal to the number of
        # unrolls (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        tf.nest.assert_same_structure(state, new_state)
        new_state = tf.nest.flatten(new_state)
        new_output = tf.nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = tf.cond(
            time >= max_sequence_length, empty_update, _maybe_copy_some_through)
    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError('Internal TensorFlow error: state and output were not '
                         'concatenated correctly.')
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]
    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for sub_state, flat_sub_state in zip(final_state, flat_state):
        sub_state.set_shape(flat_sub_state.get_shape())
    final_output = tf.nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = tf.nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)
    return final_output, final_state


def raw_rnn(cell, loop_function, parallel_iterations=None, swap_memory=False,
            scope=None):
    """Creates an `RNN` specified by RNNCell `cell` and loop function `loop_fn`.

    **NOTE: This method is still in testing, and the API may change.**

    This function is a more primitive version of `dynamic_rnn` that provides
    more direct access to the inputs each iteration.  It also provides more
    control over when to start and finish reading the sequence, and
    what to emit for the output.

    For example, it can be used to implement the dynamic decoder of a seq2seq
    model.

    Instead of working with `Tensor` objects, most operations work with
    `TensorArray` objects directly.

    The operation of `raw_rnn`, in pseudo-code, is basically the following:

    ```
    time = tf.constant(0, dtype=tf.int32)
    (finished, next_input, initial_state, _, loop_state) = loop_fn(
        time=time, cell_output=None, cell_state=None, loop_state=None)
    emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
    state = initial_state
    while not all(finished):
      (output, cell_state) = cell(next_input, state)
      (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
          time=time + 1, cell_output=output, cell_state=cell_state,
          loop_state=loop_state)
      # Emit zeros and copy forward state for minibatch entries that are finished.
      state = tf.select(finished, state, next_state)
      emit = tf.select(finished, tf.zeros_like(emit), emit)
      emit_ta = emit_ta.write(time, emit)
      # If any new minibatch entries are marked as finished, mark these
      finished = tf.logical_or(finished, next_finished)
      time += 1
    return (emit_ta, state, loop_state)
    ```

    with the additional properties that output and state may be (possibly nested)
    tuples, as determined by `cell.output_size` and `cell.state_size`, and
    as a result the final `state` and `emit_ta` may themselves be tuples.

    A simple implementation of `dynamic_rnn` via `raw_rnn` looks like this:

    ```python
    inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                            dtype=tf.float32)
    sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unpack(inputs)

    cell = tf.nn.rnn_cell.LSTMCell(num_units)

    def loop_fn(time, cell_output, cell_state, loop_state):
      emit_output = cell_output  # == None for time == 0
      if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
      else:
        next_cell_state = cell_state
      elements_finished = (time >= sequence_length)
      finished = tf.reduce_all(elements_finished)
      next_input = tf.cond(
          finished,
          lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
          lambda: inputs_ta.read(time))
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
    outputs = outputs_ta.pack()
    ```

    Args:
      cell: An instance of RNNCell.
      loop_function: A callable that takes inputs
        `(time, cell_output, cell_state, loop_state)`
        and returns the tuple
        `(finished, next_input, next_cell_state, emit_output, next_loop_state)`.
        Here `time` is an int32 scalar `Tensor`, `cell_output` is a
        `Tensor` or (possibly nested) tuple of tensors as determined by
        `cell.output_size`, and `cell_state` is a `Tensor`
        or (possibly nested) tuple of tensors, as determined by the `loop_fn`
        on its first call (and should match `cell.state_size`).
        The outputs are: `finished`, a boolean `Tensor` of
        shape `[batch_size]`, `next_input`: the next input to feed to `cell`,
        `next_cell_state`: the next state to feed to `cell`,
        and `emit_output`: the output to store for this iteration.

        Note that `emit_output` should be a `Tensor` or (possibly nested)
        tuple of tensors with shapes and structure matching `cell.output_size`
        and `cell_output` above.  The parameter `cell_state` and output
        `next_cell_state` may be either a single or (possibly nested) tuple
        of tensors.  The parameter `loop_state` and
        output `next_loop_state` may be either a single or (possibly nested) tuple
        of `Tensor` and `TensorArray` objects.  This last parameter
        may be ignored by `loop_fn` and the return value may be `None`.  If it
        is not `None`, then the `loop_state` will be propagated through the RNN
        loop, for use purely by `loop_fn` to keep track of its own state.
        The `next_loop_state` parameter returned may be `None`.

        The first call to `loop_fn` will be `time = 0`, `cell_output = None`,
        `cell_state = None`, and `loop_state = None`.  For this call:
        The `next_cell_state` value should be the value with which to initialize
        the cell's state.  It may be a final state from a previous RNN or it
        may be the output of `cell.zero_state()`.  It should be a
        (possibly nested) tuple structure of tensors.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a `TensorShape`, this must be a `Tensor` of
        appropriate type and shape `[batch_size] + cell.state_size`.
        If `cell.state_size` is a (possibly nested) tuple of ints or
        `TensorShape`, this will be a tuple having the corresponding shapes.
        The `emit_output` value may be  either `None` or a (possibly nested)
        tuple structure of tensors, e.g.,
        `(tf.zeros(shape_0, dtype=dtype_0), tf.zeros(shape_1, dtype=dtype_1))`.
        If this first `emit_output` return value is `None`,
        then the `emit_ta` result of `raw_rnn` will have the same structure and
        dtypes as `cell.output_size`.  Otherwise `emit_ta` will have the same
        structure, shapes (prepended with a `batch_size` dimension), and dtypes
        as `emit_output`.  The actual values returned for `emit_output` at this
        initializing call are ignored.  Note, this emit structure must be
        consistent across all time steps.

      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A tuple `(emit_ta, final_state, final_loop_state)` where:

      `emit_ta`: The RNN output `TensorArray`.
         If `loop_fn` returns a (possibly nested) set of Tensors for
         `emit_output` during initialization, (inputs `time = 0`,
         `cell_output = None`, and `loop_state = None`), then `emit_ta` will
         have the same structure, dtypes, and shapes as `emit_output` instead.
         If `loop_fn` returns `emit_output = None` during this call,
         the structure of `cell.output_size` is used:
         If `cell.output_size` is a (possibly nested) tuple of integers
         or `TensorShape` objects, then `emit_ta` will be a tuple having the
         same structure as `cell.output_size`, containing TensorArrays whose
         elements' shapes correspond to the shape data in `cell.output_size`.

      `final_state`: The final cell state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes.

      `final_loop_state`: The final loop state as returned by `loop_fn`.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell, or `loop_fn` is not
        a `callable`.
    """

    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
        raise TypeError('cell must be an instance of tf.nn.rnn_cell.RNNCell')
    if not callable(loop_function):
        raise TypeError('loop_function must be a callable.')
    parallel_iterations = parallel_iterations or 32

    # Create a new scope in which the caching device is either determined by
    # the parent scope, or is set to place the cached Variable using the same
    # placement as for the rest of the RNN.
    with tf.variable_scope(scope or 'RNN') as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        time = tf.constant(0, dtype=tf.int32)
        finished, next_input, initial_state, emit_structure, init_loop_state = \
            loop_function(time, None, None, None)
        flat_input = tf.nest.flatten(next_input)

        # We need a surrogate loop state for the while loop
        loop_state = (init_loop_state if init_loop_state is not None
                      else tf.constant(0, dtype=tf.int32))

        # Shape validation for inputs
        input_shapes = [fi.get_shape() for fi in flat_input]
        static_batch_size = input_shapes[0][0]
        for input_shape in input_shapes:
            static_batch_size.merge_with(input_shape[0])
        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = tf.shape(flat_input[0])[0]

        # Shape validation for state
        tf.nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = tf.nest.flatten(state)
        flat_state = [tf.convert_to_tensor(s) for s in flat_state]
        state = tf.nest.pack_sequence_as(
            structure=state, flat_sequence=flat_state)

        if emit_structure is not None:
            flat_emit_structure = tf.nest.flatten(emit_structure)
            flat_emit_size = [emit.get_shape() for emit in flat_emit_structure]
            flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
        else:
            emit_structure = cell.output_size
            flat_emit_size = tf.nest.flatten(emit_structure)
            flat_emit_dtypes = [flat_state[0].dtype] * len(flat_emit_size)

        flat_emit_ta = [tf.TensorArray(
            dtype=dtype, dynamic_size=True, size=0,
            name='rnn_output_%d' % i)
                        for i, dtype in enumerate(flat_emit_dtypes)]
        emit_ta = tf.nest.pack_sequence_as(
            structure=emit_structure, flat_sequence=flat_emit_ta)
        flat_zero_emit = [tf.zeros(
            tf.nn.rnn_cell._state_size_with_prefix(size, prefix=[batch_size]),
            dtype) for size, dtype in zip(flat_emit_size, flat_emit_dtypes)]
        zero_emit = tf.nest.pack_sequence_as(
            structure=emit_structure, flat_sequence=flat_zero_emit)

        def _cond(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def _body(time, finished, current_input, emit_ta, state, loop_state):
            next_output, cell_state = cell(current_input, state)

            tf.nest.assert_same_structure(state, cell_state)
            tf.nest.assert_same_structure(cell.output_size, next_output)
            next_time = time + 1

            next_finished, next_input, next_state, emit_output, \
            next_loop_state = loop_function(
                next_time, next_output, cell_state, loop_state)

            tf.nest.assert_same_structure(state, next_state)
            tf.nest.assert_same_structure(current_input, next_input)
            tf.nest.assert_same_structure(emit_ta, emit_output)

            loop_state = loop_state if next_loop_state is None \
                else next_loop_state

            def _copy_some_through(current, candidate):
                current_flat = tf.nest.flatten(current)
                candidate_flat = tf.nest.flatten(candidate)
                result_flat = [tf.select(finished, curr, cand)
                               for curr, cand
                               in zip(current_flat, candidate_flat)]
                return tf.nest.pack_sequence_as(
                    structure=current, flat_sequence=result_flat)

            emit_output = _copy_some_through(zero_emit, emit_output)
            emit_output_flat = tf.nest.flatten(emit_output)
            finished = tf.logical_or(finished, next_finished)
            next_state = _copy_some_through(state, next_state)
            emit_ta_flat = tf.nest.flatten(emit_ta)
            emit_ta_flat = [ta.write(time, emit)
                            for ta, emit
                            in zip(emit_ta_flat, emit_output_flat)]
            emit_ta = tf.nest.pack_sequence_as(
                structure=emit_structure, flat_sequence=emit_ta_flat)
            return next_time, finished, next_input, emit_ta, next_state, \
                loop_state

        returned = tf.while_loop(
            cond=_cond, body=_body,
            loop_vars=[time, finished, next_input, emit_ta, state, loop_state],
            parallel_iterations=parallel_iterations, swap_memory=swap_memory)
        emit_ta, final_state, final_loop_state = returned[-3:]
        if init_loop_state is None:
            final_loop_state = None
        return emit_ta, final_state, final_loop_state
