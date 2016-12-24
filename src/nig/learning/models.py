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

import abc
import inspect
import itertools
import logging
import numpy as np
import tensorflow as tf

from six import with_metaclass

from . import metrics
from ..utilities.tensorflow import graph_context, copy_op_to_graph, \
    copy_variable_to_graph

__author__ = 'eaplatanios'

__all__ = ['Model', 'CombinedModel', 'LinearCombinationModel']

__TENSORS_DIFFERENT_GRAPHS_ERROR__ = 'All tensors should be in the same graph.'
__SUPPORTED_INTERNAL_OPTIMIZER_OPTS__ = {
    'batch_size', 'max_iter', 'abs_loss_chg_tol', 'rel_loss_chg_tol',
    'loss_chg_iter_below_tol', 'grads_processor'}

logger = logging.getLogger(__name__)


class Model(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, inputs, outputs, train_outputs=None, loss=None,
                 loss_summary=False, gradients=None, optimizer=None,
                 optimizer_opts=None, graph=None):
        """

        If `loss` is `None` and `gradients` is provided, then `loss` is set to
        the sum of the norms of the gradients.

        Args:
            inputs:
            outputs:
            train_outputs:
            loss:
            loss_summary:
            gradients:
            optimizer:
            optimizer_opts:
            graph:
        """
        if graph is not None:
            self.graph = graph
        if isinstance(inputs, list):
            if graph is None:
                self.graph = inputs[0].graph
            if any(i.graph != self.graph for i in inputs):
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        elif isinstance(inputs, dict):
            input_values = list(inputs.values())
            if graph is None:
                self.graph = input_values[0].graph
            if any(i.graph != self.graph for i in input_values):
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        elif graph is None:
            self.graph = inputs.graph
        else:
            if inputs.graph != self.graph:
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        if isinstance(outputs, list):
            if any(o.graph != self.graph for o in outputs):
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        elif isinstance(outputs, dict):
            output_values = list(outputs.values())
            if any(o.graph != self.graph for o in output_values):
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        else:
            if outputs.graph != self.graph:
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        self.inputs = inputs
        self.outputs = outputs
        if isinstance(outputs, list) and isinstance(loss, list):
            if len(outputs) != len(loss):
                raise ValueError('The number of provided output ops must match '
                                 'the number of provided loss ops.')
        self.trainable = (loss is not None or gradients is not None) \
            and optimizer is not None
        self.train_outputs = train_outputs
        self.loss = loss
        self.loss_op = None
        self.loss_summary = loss_summary
        self._gradients = gradients
        self.uses_external_optimizer = inspect.isclass(optimizer)
        if self.uses_external_optimizer and gradients is not None:
            raise ValueError('Cannot use an external optimizer with the '
                             'gradients being provided.')
        self.provided_optimizer = optimizer
        self.optimizer = None
        self.optimizer_opts = dict() if optimizer_opts is None \
            else optimizer_opts
        self.train_op = None
        if self.trainable:
            self._process_train_args()

    @graph_context
    def _process_train_args(self):
        # Process train_outputs
        if self.train_outputs is not None:
            if isinstance(self.train_outputs, list):
                if any(o.graph != self.graph for o in self.train_outputs):
                    raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
            elif isinstance(self.train_outputs, dict):
                output_values = list(self.train_outputs.values())
                if any(o.graph != self.graph for o in output_values):
                    raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
            elif self.train_outputs.graph != self.graph:
                raise ValueError(__TENSORS_DIFFERENT_GRAPHS_ERROR__)
        elif isinstance(self.outputs, list):
            self.train_outputs = [tf.placeholder(
                dtype=output.dtype, shape=output.get_shape(),
                name=output.name.split(':')[0] + '/observed')
                                  for output in self.outputs]
        elif isinstance(self.outputs, dict):
            self.train_outputs = {k: tf.placeholder(
                dtype=v.dtype, shape=v.get_shape(),
                name=v.name.split(':')[0] + '/observed')
                                  for k, v in self.outputs.items()}
        else:
            self.train_outputs = tf.placeholder(
                dtype=self.outputs.dtype, shape=self.outputs.get_shape(),
                name=self.outputs.name.split(':')[0] + '/observed')
        # Process loss
        self.update_loss(loss=self.loss)

    def update_loss(self, loss):
        if not isinstance(loss, metrics.Metric):
            raise TypeError('Unsupported loss type %s encountered. The loss '
                            'is required to be a NIG Metric.' % type(loss))
        self.loss = loss
        self.loss_op = self.loss(self.outputs, self.train_outputs)
        if self.loss_summary:
            tf.summary.scalar(name=self.loss_op.op.name, tensor=self.loss_op)

        # Process optimizer and optimizer_opts
        if self.uses_external_optimizer:
            with tf.name_scope('external_optimizer'):
                if 'options' in self.optimizer_opts:
                    if 'disp' not in self.optimizer_opts['options']:
                        self.optimizer_opts['options']['disp'] = False
                else:
                    self.optimizer_opts['options'] = {'disp': False}
                self.optimizer = self.provided_optimizer(
                    self.loss_op, **self.optimizer_opts)
        elif not callable(self.provided_optimizer):
            raise TypeError('Unsupported optimizer type %s encountered.'
                            % type(self.provided_optimizer))
        else:
            with tf.name_scope('optimizer'):
                self.optimizer = self.provided_optimizer()
            provided_opts = set(self.optimizer_opts.keys())
            unsupported = provided_opts - __SUPPORTED_INTERNAL_OPTIMIZER_OPTS__
            if len(unsupported) > 0:
                logger.warn('Ignoring unsupported optimizer options %s. '
                            'Supported options are %s.'
                            % (unsupported,
                               __SUPPORTED_INTERNAL_OPTIMIZER_OPTS__))
            grads_processor = self.optimizer_opts.get('grads_processor', None)
            if isinstance(self._gradients, dict):
                self._gradients = [(v, k) for k, v in self._gradients.items()]
            if grads_processor is not None and self._gradients is None:
                if not callable(grads_processor):
                    raise TypeError('Unsupported gradients processor type %s '
                                    'encountered. The provided gradients '
                                    'processor needs to be a function.'
                                    % type(grads_processor))
                trainable_vars = tf.trainable_variables()
                grads = tf.gradients(ys=self.loss_op, xs=trainable_vars)
                grads = grads_processor(grads)
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=zip(grads, trainable_vars))
            elif self._gradients is not None:
                if grads_processor is not None:
                    self._gradients = [(grads_processor(g), v)
                                       for g, v in self._gradients]
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=self._gradients)
            else:
                self.train_op = self.optimizer.minimize(loss=self.loss_op)

    @staticmethod
    def _add_to_feed_dict(feed_dict, new_feed_dict):
        if feed_dict is not None:
            for key, value in new_feed_dict.items():
                if key in feed_dict:
                    value = np.concatenate((feed_dict[key], value), axis=0)
                feed_dict.update({key: value})
            return feed_dict
        return new_feed_dict

    def get_feed_dict(self, data, is_train=False, feed_dict=None):
        if isinstance(data, np.ndarray):
            return self._add_to_feed_dict(feed_dict, {self.inputs: data})
        if isinstance(data, list) or isinstance(data, tuple):
            tensors = []
            if isinstance(self.inputs, list):
                tensors.extend(self.inputs)
            elif isinstance(self.inputs, dict):
                raise TypeError('Data should be provided as dictionaries '
                                'when model variables are represented as '
                                'dictionaries.')
            else:
                tensors.append(self.inputs)
            if is_train and self.train_outputs is not None:
                if isinstance(self.train_outputs, list):
                    tensors.extend(self.train_outputs)
                elif isinstance(self.train_outputs, dict):
                    raise TypeError('Data should be provided as dictionaries '
                                    'when model variables are represented as '
                                    'dictionaries.')
                else:
                    tensors.append(self.train_outputs)
            return self._add_to_feed_dict(feed_dict, dict(zip(tensors, data)))
        if isinstance(data, dict):
            new_feed_dict = {Model._get_tensor_name(k): v
                             for k, v in data.items()}
            return self._add_to_feed_dict(feed_dict, new_feed_dict)
        if not is_train or self.train_outputs is None:
            return self._add_to_feed_dict(feed_dict, {self.inputs: data})
        new_feed_dict = {self.inputs: data[0], self.train_outputs: data[1]}
        return self._add_to_feed_dict(feed_dict, new_feed_dict)

    @staticmethod
    def _get_tensor_name(tensor):
        if isinstance(tensor, str):
            return tensor
        return tensor.name

    def copy_to_graph(self, graph, variables=None, scope=None):
        if graph == self.graph and scope is None:
            return self
        variables = [] if variables is None else variables
        for variable in self._variables():
            variables.append(copy_variable_to_graph(
                org_instance=variable, to_graph=graph, scope=scope))
        inputs = self._copy_ops_to_graph(
                ops=self.inputs, graph=graph, variables=variables, scope=scope)
        outputs = self._copy_ops_to_graph(
            ops=self.outputs, graph=graph, variables=variables, scope=scope)
        if self.trainable:
            if self.train_outputs is not None:
                train_outputs = self._copy_ops_to_graph(
                    ops=self.train_outputs, graph=graph, variables=variables,
                    scope=scope)
            else:
                train_outputs = None
            return Model(
                inputs=inputs, outputs=outputs, train_outputs=train_outputs,
                loss=self.loss, optimizer=self.provided_optimizer,
                optimizer_opts=self.optimizer_opts)
        return Model(inputs=inputs, outputs=outputs)

    def _variables(self):
        if self.trainable:
            start_ops = self.loss_op
        elif isinstance(self.outputs, dict):
            start_ops = self.outputs.values()
        else:
            start_ops = self.outputs
        return self._op_variables(start_ops)

    @graph_context
    def _op_variables(self, ops, traversed_ops=None, tf_variables=None):
        if tf_variables is None:
            tf_variables = {v.name.split(':')[0]: v
                            for v in tf.global_variables()}
        if traversed_ops is None:
            traversed_ops = set()
        if isinstance(ops, list):
            return set(itertools.chain.from_iterable(
                self._op_variables(op, traversed_ops, tf_variables)
                for op in ops))
        traversed_ops.add(ops)
        if isinstance(ops, tf.Operation):
            variables = set(itertools.chain(
                *[self._op_variables(input_op, traversed_ops, tf_variables)
                  for input_op in list(ops.inputs) + list(ops.control_inputs)
                  if input_op not in traversed_ops]))
        elif isinstance(ops, tf.Tensor):
            variables = set()
            if ops.op.type == 'Variable':
                variables.add(tf_variables[ops.op.name])
            input_ops = list(ops.op.inputs) + list(ops.op.control_inputs)
            if len(input_ops) > 0:
                variables.update(set(itertools.chain(
                    *[self._op_variables(input_op, traversed_ops, tf_variables)
                      for input_op in input_ops
                      if input_op not in traversed_ops])))
        else:
            raise TypeError('Invalid op type provided.')
        return variables

    @staticmethod
    def _copy_ops_to_graph(ops, graph, variables=None, scope=None):
        if variables is None:
            variables = []
        if isinstance(ops, list):
            return [copy_op_to_graph(
                org_instance=op, to_graph=graph, variables=variables,
                copy_summaries=True, scope=scope) for op in ops]
        if isinstance(ops, dict):
            return {name: copy_op_to_graph(
                org_instance=op, to_graph=graph, variables=variables,
                copy_summaries=True, scope=scope) for name, op in ops.items()}
        return copy_op_to_graph(
            org_instance=ops, to_graph=graph, variables=variables,
            copy_summaries=True, scope=scope)


class CombinedModel(Model):
    def __init__(self, models, combined_outputs=None, graph=None,
                 copy_models=False):
        if graph is None and copy_models:
            self.graph = tf.Graph()
        elif graph is None:
            self.graph = models[0].graph
        else:
            self.graph = graph
        if graph is None and not copy_models \
                and any(model.graph != self.graph for model in models):
            raise ValueError('All models are required to lie in the same '
                             'graph, when "copy_models" is set to "False".')
        if copy_models:
            self.models = [model.copy_to_graph(graph=self.graph)
                           for model in models]
        else:
            self.models = models
        inputs = CombinedModel._combine_model_variables(self.models, 'inputs')
        super(CombinedModel, self).__init__(
            inputs=inputs, outputs=combined_outputs,
            train_outputs=self.models[0].train_outputs, loss=None,
            loss_summary=False, optimizer=None, optimizer_opts=None,
            graph=self.graph)

    @staticmethod
    def _combine_model_variables(models, variables_name):
        variables = [getattr(model, variables_name) for model in models]
        if isinstance(variables[0], list):
            return [v for model_vars in variables for v in model_vars]
        if isinstance(variables[0], dict):
            return {'model_%d/%s' % (m, n): v
                    for m, model_vars in enumerate(variables)
                    for n, v in model_vars.items()}
        return variables

    def get_feed_dict(self, data, is_train=False, feed_dict=None):
        # TODO: Update this to use the feed_dict argument.
        feed_dict = dict()
        for m, model in enumerate(self.models):
            model_data = data
            if isinstance(model_data, dict):
                model_data = {n if not isinstance(n, str)
                              else 'model_%d/%s' % (m, n): t
                              for n, t in model_data.items()}
            feed_dict.update(model.get_feed_dict(model_data, is_train=is_train))
        return feed_dict


class LinearCombinationModel(Model):
    def __init__(self, models, weights=None, loss_summary=False, graph=None,
                 copy_models=False):
        if graph is None and copy_models:
            self.graph = tf.Graph()
        elif graph is None:
            self.graph = models[0].graph
        else:
            self.graph = graph
        if graph is None and not copy_models \
                and any(model.graph != self.graph for model in models):
            raise ValueError('All models are required to lie in the same '
                             'graph, when "copy_models" is set to "False".')
        if copy_models:
            self.models = [model.copy_to_graph(graph=self.graph)
                           for model in models]
        else:
            self.models = models
        inputs = LinearCombinationModel._combine_model_variables(
            self.models, 'inputs')
        outputs = LinearCombinationModel._combine_model_variables(
            self.models, 'outputs')
        super(LinearCombinationModel, self).__init__(
            inputs=inputs, outputs=outputs,  train_outputs=None, loss=None,
            loss_summary=False, optimizer=None, optimizer_opts=None,
            graph=self.graph)
        with tf.name_scope('combination'):
            self.outputs = tf.pack(
                [model.outputs for model in self.models], axis=-1)
            if weights is None:
                self.weights = tf.Variable(
                    initial_value=np.ones(len(self.models)) / len(self.models),
                    trainable=False)
            else:
                self.weights, _ = self._copy_op_to_graph(
                    op=weights, graph=self.graph)
            # TODO: Add support for other output formats.
            self.outputs = tf.mul(self.weights, self.outputs)
            self.outputs = tf.reduce_sum(self.outputs, reduction_indices=[-1])
            self.train_outputs = self.models[0].train_outputs
            # weights = tf.unpack(self.weights)
            # self.outputs = tf.add_n(
            #     inputs=[weight * output
            #             for weight, output in zip(weights, self.outputs)])
            # self.trainable = all(model.trainable for model in self.models)
            # if self.trainable:
            #     self.train_outputs = CombinedModel._combine_model_variables(
            #         models=self.models, variables_name='train_outputs')
            # TODO: Make this a trainable model.
            self.trainable = False
            self.loss_op = tf.add_n([model.loss_op for model in self.models])
            self.loss_op /= len(self.models)
            self.loss_summary = loss_summary
            if self.loss_summary:
                tf.summary.scalar(
                    name=self.loss_op.op.name, tensor=self.loss_op)

    @staticmethod
    def _combine_model_variables(models, variables_name):
        variables = [getattr(model, variables_name) for model in models]
        if isinstance(variables[0], list):
            return [v for model_vars in variables for v in model_vars]
        if isinstance(variables[0], dict):
            return {'model_%d/%s' % (m, n): v
                    for m, model_vars in enumerate(variables)
                    for n, v in model_vars.items()}
        return variables

    def get_feed_dict(self, data, is_train=False, feed_dict=None):
        # TODO: Update this to use the feed_dict argument.
        feed_dict = dict()
        for m, model in enumerate(self.models):
            model_data = data
            if isinstance(model_data, dict):
                model_data = {n if not isinstance(n, str)
                              else 'model_%d/%s' % (m, n): t
                              for n, t in model_data.items()}
            feed_dict.update(model.get_feed_dict(model_data, is_train=is_train))
        return feed_dict

    @graph_context
    def copy_to_graph(self, graph, variables=None, scope=None):
        if graph == self.graph and scope is None:
            return self
        weights, variables = self._copy_op_to_graph(
            op=self.weights, graph=graph, variables=variables, scope=scope)
        models = [model.copy_to_graph(
            graph=graph, variables=variables, scope=scope)
                  for model in self.models]
        return LinearCombinationModel(
            models=models, weights=weights, graph=graph, copy_models=False)

    def _copy_op_to_graph(self, op, graph, variables=None, scope=None):
        # TODO: Factor into Model._copy_op_to_graph.
        if graph == self.graph and scope is None:
            return op, variables
        if graph == self.graph:
            print(scope)
        variables = [] if variables is None else variables
        variables.extend([copy_variable_to_graph(
            org_instance=variable, to_graph=graph, scope=scope)
                          for variable in self._op_variables(op)])
        op = Model._copy_ops_to_graph(
            ops=op, graph=graph, variables=variables, scope=scope)
        return op, variables
