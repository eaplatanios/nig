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

__author__ = 'eaplatanios'

__all__ = ['graph_context', 'name_scope_context', 'copy_variable_to_graph',
           'copy_op_to_graph', 'check_if_present', 'get_copied_op']


def graph_context(func):
    def func_wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return func(self, *args, **kwargs)
    return func_wrapper


def name_scope_context(func):
    def func_wrapper(self, *args, **kwargs):
        if self.name_scope is not None:
            with tf.name_scope(self.name_scope):
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)
    return func_wrapper


# TODO: Clean this up...A LOT...

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""## Functions for copying elements from one graph to another.

These functions allow for recursive copying of elements (ops and variables)
from one graph to another. The copied elements are initialized inside a
user-specified scope in the other graph. There are separate functions to
copy ops and variables.
There is also a function to retrive the copied version of an op from the
first graph inside a scope in the second graph.

@@copy_op_to_graph
@@copy_variable_to_graph
@@get_copied_op
"""

import itertools
import tensorflow as tf

from copy import deepcopy
from tensorflow.python.ops.variables import Variable
from tensorflow.python.client.session import Session
from tensorflow.python.framework import ops


def copy_variable_to_graph(org_instance, to_graph, scope=None):
    """Given a `Variable` instance from one `Graph`, initializes and returns
    a copy of it from another `Graph`, under the specified scope
    (default `""`).

    Args:
    org_instance: A `Variable` from some `Graph`.
    to_graph: The `Graph` to copy the `Variable` to.
    scope: A scope for the new `Variable` (default `""`).

    Returns:
        The copied `Variable` from `to_graph`.

    Raises:
        TypeError: If `org_instance` is not a `Variable`.
    """

    if not isinstance(org_instance, Variable):
        raise TypeError(str(org_instance) + " is not a Variable")

    #The name of the new variable
    if scope is not None:
        new_name = (scope + '/' +
                    org_instance.name[:org_instance.name.index(':')])
    else:
        new_name = org_instance.name[:org_instance.name.index(':')]

    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given scope,
    #except the special ones required for variable initialization and
    #training.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if (name == ops.GraphKeys.GLOBAL_VARIABLES or
                        name == ops.GraphKeys.TRAINABLE_VARIABLES or
                        scope is None):
                collections.append(name)
            else:
                collections.append(scope + '/' + name)

    #See if its trainable.
    trainable = (org_instance in org_instance.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES))
    #Get the initial value
    with org_instance.graph.as_default():
        temp_session = Session()
        init_value = temp_session.run(org_instance.initialized_value())

    #Initialize the new variable
    with to_graph.as_default():
        new_var = Variable(init_value,
                           trainable,
                           name=new_name,
                           collections=collections,
                           validate_shape=True)

    return new_var


def copy_op_to_graph(org_instance, to_graph, variables, copy_summaries=False,
                     scope=None):
    """Given an `Operation` 'org_instance` from one `Graph`,
    initializes and returns a copy of it from another `Graph`,
    under the specified scope (default `""`).

    The copying is done recursively, so any `Operation` whose output
    is required to evaluate the `org_instance`, is also copied (unless
    already done).

    Since `Variable` instances are copied separately, those required
    to evaluate `org_instance` must be provided as input.

    Args:
    org_instance: An `Operation` from some `Graph`. Could be a
        `Placeholder` as well.
    to_graph: The `Graph` to copy `org_instance` to.
    variables: An iterable of `Variable` instances to copy `org_instance` to.
    scope: A scope for the new `Variable` (default `""`).

    Returns:
        The copied `Operation` from `to_graph`.

    Raises:
        TypeError: If `org_instance` is not an `Operation` or `Tensor`.
    """
    #The name of the new instance
    if scope is not None:
        new_name = scope + '/' + org_instance.name
    else:
        new_name = org_instance.name

    #Extract names of variables
    copied_variables = dict((x.name, x) for x in variables)

    #If a variable by the new name already exists, return the
    #correspondng tensor that will act as an input
    if new_name in copied_variables:
        op = to_graph.get_tensor_by_name(copied_variables[new_name].name)
        if copy_summaries:
            _copy_summaries(
                op=org_instance, to_graph=to_graph, variables=variables,
                scope=scope)
        return op

    #If an instance of the same name exists, return appropriately
    already_present = check_if_present(op_name=new_name, to_graph=to_graph)
    if already_present is not None:
        if copy_summaries:
            _copy_summaries(
                op=org_instance, to_graph=to_graph, variables=variables,
                scope=scope)
        return already_present

    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given scope.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if scope is None:
                collections.append(name)
            else:
                collections.append(scope + '/' + name)

    #Take action based on the class of the instance

    if isinstance(org_instance, ops.Tensor):

        #If its a Tensor, it is one of the outputs of the underlying
        #op. Therefore, copy the op itself and return the appropriate
        #output.
        op = org_instance.op
        new_op = copy_op_to_graph(
            org_instance=op, to_graph=to_graph, variables=variables,
            copy_summaries=copy_summaries, scope=scope)
        output_index = op.outputs.index(org_instance)
        new_tensor = new_op.outputs[output_index]
        new_tensor.set_shape(org_instance.get_shape())
        #Add to collections if any
        for collection in collections:
            to_graph.add_to_collection(collection, new_tensor)

        if copy_summaries:
            _copy_summaries(
                op=org_instance, to_graph=to_graph, variables=variables,
                scope=scope)

        return new_tensor

    elif isinstance(org_instance, ops.Operation):

        op = org_instance

        #If it has an original_op parameter, copy it
        if op._original_op is not None:
            new_original_op = copy_op_to_graph(
                org_instance=op._original_op, to_graph=to_graph,
                variables=variables, copy_summaries=copy_summaries, scope=scope)
        else:
            new_original_op = None

        #If it has control inputs, call this function recursively on each.
        new_control_inputs = [copy_op_to_graph(
            org_instance=x, to_graph=to_graph, variables=variables,
            copy_summaries=copy_summaries, scope=scope)
                              for x in op.control_inputs]

        already_present = check_if_present(op_name=new_name, to_graph=to_graph)
        if already_present is not None:
            return already_present

        #Make a new node_def based on that of the original.
        #An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
        #stores String-based info such as name, device and type of the op.
        #Unique to every Operation instance.
        new_node_def = deepcopy(op._node_def)
        #Change the name
        new_node_def.name = new_name

        #Copy the other inputs needed for initialization
        output_types = op._output_types[:]
        input_types = op._input_types[:]

        #Make a copy of the op_def too.
        #Its unique to every _type_ of Operation.
        op_def = deepcopy(op._op_def)

        #Initialize a new Operation instance
        new_op = ops.Operation(new_node_def,
                               to_graph,
                               None,
                               output_types,
                               new_control_inputs,
                               None,
                               new_original_op,
                               op_def)
        #Use Graph's hidden methods to add the op
        to_graph._add_op(new_op)
        # to_graph._record_op_seen_by_control_dependencies(new_op)

        #If it has inputs, call this function recursively on each.
        for x in op.inputs:
            new_op._add_input(copy_op_to_graph(
                org_instance=x, to_graph=to_graph, variables=variables,
                copy_summaries=copy_summaries, scope=scope))

        to_graph._record_op_seen_by_control_dependencies(new_op)

        for device_function in reversed(to_graph._device_function_stack):
            new_op._set_device(device_function(new_op))

        return new_op

    else:
        raise TypeError("Could not copy instance: " + str(org_instance))


def check_if_present(op_name, to_graph):
    try:
        op = to_graph.as_graph_element(
            op_name, allow_tensor=True, allow_operation=True)
        return op
    except:
        return None


def _copy_summaries(op, to_graph, variables, scope):
    # TODO: This is kind of hacky right now.
    if isinstance(op, tf.Tensor):
        for consumer in op.consumers():
            if 'Summary' in consumer.type:
                for output in consumer.outputs:
                    copy_op_to_graph(
                        org_instance=output, to_graph=to_graph,
                        variables=variables, copy_summaries=True, scope=scope)


def get_copied_op(org_instance, graph, scope=None):
    """Given an `Operation` instance from some `Graph`, returns
    its namesake from `graph`, under the specified scope
    (default `""`).

    If a copy of `org_instance` is present in `graph` under the given
    `scope`, it will be returned.

    Args:
    org_instance: An `Operation` from some `Graph`.
    graph: The `Graph` to be searched for a copr of `org_instance`.
    scope: The scope `org_instance` is present in.

    Returns:
        The `Operation` copy from `graph`.
    """

    #The name of the copied instance
    if scope is not None:
        new_name = scope + '/' + org_instance.name
    else:
        new_name = org_instance.name

    return graph.as_graph_element(new_name, allow_tensor=True,
                                  allow_operation=True)
