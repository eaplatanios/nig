import abc
import inspect
import itertools
import numpy as np
import tensorflow as tf

from six import with_metaclass

from ..utilities.tensorflow import graph_context, copy_op_to_graph, \
    copy_variable_to_graph

__author__ = 'eaplatanios'


class Model(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, inputs, outputs, train_outputs=None, loss=None,
                 loss_summary=False, optimizer=None, optimizer_opts=None,
                 train_op=None):
        if isinstance(inputs, list):
            self.graph = inputs[0].graph
        else:
            self.graph = inputs.graph
        self.inputs = inputs
        self.outputs = outputs
        if isinstance(outputs, list) and isinstance(loss, list):
            if len(outputs) != len(loss):
                raise ValueError('The number of provided output ops must match '
                                 'the number of provided loss ops.')
        self.trainable = (loss is not None and optimizer is not None) \
            or train_op is not None
        if self.trainable:
            self._process_train_args(
                train_outputs=train_outputs, loss=loss,
                loss_summary=loss_summary, optimizer=optimizer,
                optimizer_opts=optimizer_opts, train_op=train_op)

    @graph_context
    def _process_train_args(self, train_outputs, loss, loss_summary, optimizer,
                            optimizer_opts, train_op):
        # Process train_outputs
        if train_outputs is not None:
            self.train_outputs = train_outputs
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
        if callable(loss):
            loss = loss(self.outputs, self.train_outputs)
        if not isinstance(loss, tf.Tensor):
            raise TypeError('Unsupported loss type %s encountered.'
                            % type(loss))
        self.loss = loss
        self.loss_summary = loss_summary
        if loss_summary:
            tf.scalar_summary(self.loss.op.name, self.loss)
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        # Process train_op
        self.uses_external_optimizer = inspect.isclass(optimizer)
        self.provided_optimizer = optimizer
        self.optimizer_opts = optimizer_opts
        if train_op is not None:
            if callable(train_op):
                self.train_op = train_op()
            elif not isinstance(train_op, tf.Operation):
                raise TypeError('Unsupported train op type %s '
                                'encountered,' % type(train_op))
            else:
                self.train_op = train_op
            return

        # Process optimizer and optimizer_opts
        if self.uses_external_optimizer:
            with tf.name_scope('external_optimizer'):
                if 'options' in optimizer_opts:
                    if 'disp' not in optimizer_opts['options']:
                        optimizer_opts['options']['disp'] = False
                else:
                    optimizer_opts['options'] = {'disp': False}
                self.optimizer = optimizer(self.loss, **optimizer_opts)
        elif not isinstance(optimizer, tf.train.Optimizer):
            raise TypeError('Unsupported optimizer type %s encountered.'
                            % type(optimizer))
        else:
            self.optimizer = optimizer
            grads_processor = optimizer_opts.get('grads_processor', None)
            if grads_processor is not None:
                trainable_vars = tf.trainable_variables()
                grads = tf.gradients(ys=self.loss, xs=trainable_vars)
                grads = grads_processor(grads)
                self.train_op = optimizer.apply_gradients(
                    grads_and_vars=zip(grads, trainable_vars),
                    global_step=self.global_step)
            else:
                self.train_op = optimizer.minimize(
                    loss=self.loss, global_step=self.global_step)

    def get_feed_dict(self, data, is_train=False):
        if isinstance(data, np.ndarray):
            return {self.inputs: data}
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
            if is_train:
                if isinstance(self.train_outputs, list):
                    tensors.extend(self.train_outputs)
                elif isinstance(self.train_outputs, dict):
                    raise TypeError('Data should be provided as dictionaries '
                                    'when model variables are represented as '
                                    'dictionaries.')
                else:
                    tensors.append(self.train_outputs)
            return dict(zip(tensors, data))
        if isinstance(data, dict):
            return {Model._get_tensor_name(k): v for k, v in data.items()}
        if not is_train:
            return {self.inputs: data}
        return {self.inputs: data[0], self.train_outputs: data[1]}

    @staticmethod
    def _get_tensor_name(tensor):
        if isinstance(tensor, str):
            return tensor
        return tensor.name

    def update_loss(self, loss):
        self._process_train_args(
            train_outputs=self.train_outputs, loss=loss,
            loss_summary=self.loss_summary, optimizer=self.provided_optimizer,
            optimizer_opts=self.optimizer_opts, train_op=None)

    def copy_to_graph(self, graph, scope=''):
        variables = []
        for variable in self._variables():
            variables.append(copy_variable_to_graph(
                org_instance=variable, to_graph=graph, scope=scope))
        inputs = self._copy_ops_to_graph(
                ops=self.inputs, graph=graph, variables=variables, scope=scope)
        outputs = self._copy_ops_to_graph(
            ops=self.outputs, graph=graph, variables=variables, scope=scope)
        if self.trainable:
            train_outputs = self._copy_ops_to_graph(
                ops=self.train_outputs, graph=graph, variables=variables,
                scope=scope)
            loss = self._copy_ops_to_graph(
                ops=self.loss, graph=graph, variables=variables, scope=scope)
            if self.uses_external_optimizer:
                return Model(
                    inputs=inputs, outputs=outputs, train_outputs=train_outputs,
                    loss=loss, optimizer=self.provided_optimizer,
                    optimizer_opts=self.optimizer_opts)

            train_op = self._copy_ops_to_graph(
                ops=self.train_op, graph=graph, variables=variables,
                scope=scope)
            return Model(
                inputs=inputs, outputs=outputs, train_outputs=train_outputs,
                loss=loss, optimizer_opts=self.optimizer_opts,
                train_op=train_op)
        return Model(inputs=inputs, outputs=outputs)

    def _variables(self):
        all_variables = {var.name.split(':')[0]: var
                         for var in tf.all_variables()}
        if self.trainable and self.uses_external_optimizer:
            start_ops = self.loss
        elif self.trainable:
            start_ops = self.train_op
        elif isinstance(self.outputs, dict):
            start_ops = self.outputs.values()
        else:
            start_ops = self.outputs
        if isinstance(start_ops, list):
            return set(all_variables[var]
                       for var in itertools.chain.from_iterable(
                self._op_variables(output) for output in start_ops))
        return set(all_variables[var] for var in self._op_variables(start_ops))

    def _op_variables(self, op, traversed_ops=None):
        if traversed_ops is None:
            traversed_ops = set()
        traversed_ops.add(op)
        if isinstance(op, tf.Operation):
            return set(itertools.chain(
                *[self._op_variables(input_op, traversed_ops)
                  for input_op in list(op.inputs) + list(op.control_inputs)
                  if input_op not in traversed_ops]))
        elif isinstance(op, tf.Tensor):
            variables = set()
            if op.op.type == 'Variable':
                variables.add(op.op.name)
            input_ops = list(op.op.inputs) + list(op.op.control_inputs)
            if len(input_ops) > 0:
                variables.update(set(itertools.chain(
                    *[self._op_variables(input_op, traversed_ops)
                      for input_op in input_ops
                      if input_op not in traversed_ops])))
            return variables
        raise TypeError('Invalid op type provided.')

    @staticmethod
    def _copy_ops_to_graph(ops, graph, variables=None, scope=''):
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
