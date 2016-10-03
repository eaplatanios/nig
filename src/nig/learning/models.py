import abc
import inspect
import itertools
import numpy as np
import tensorflow as tf
from six import with_metaclass

from nig.utilities.generic import logger, raise_error
from nig.utilities.tensorflow import copy_op_to_graph, copy_variable_to_graph

__author__ = 'eaplatanios'


class Model(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, inputs, outputs, train_outputs=None, loss=None,
                 loss_summary=False, optimizer=None, optimizer_opts=None,
                 grads_processor=None, train_op=None):
        if isinstance(inputs, list):
            self.graph = inputs[0].graph
        else:
            self.graph = inputs.graph
        self.inputs = inputs
        self.outputs = outputs
        if isinstance(outputs, list) and isinstance(loss, list):
            if len(outputs) != len(loss):
                raise_error(ValueError, 'The number of provided output ops '
                                        'must match the number of provided '
                                        'loss ops.')
        self.trainable = (loss is not None and optimizer is not None) \
            or train_op is not None
        if self.trainable:
            with self.graph.as_default():
                self.train_outputs = self._process_train_outputs(train_outputs)
                self.loss = self._process_loss(loss)
                if loss_summary:
                    tf.scalar_summary(self.loss.op.name, self.loss)
                self.global_step = \
                    tf.contrib.framework.get_or_create_global_step()
                self.uses_external_optimizer = inspect.isclass(optimizer)
                self.optimizer_opts = optimizer_opts
                if self.uses_external_optimizer:
                    self._optimizer = optimizer
                    self.optimizer = self._process_optimizer(
                        self._optimizer, self.optimizer_opts)
                else:
                    optimizer = self._process_optimizer(optimizer, None)
                    self.train_op = self._train_op(
                        train_op=train_op, optimizer=optimizer,
                        grads_processor=grads_processor)

    def _process_train_outputs(self, train_outputs):
        if train_outputs is not None:
            return train_outputs
        if isinstance(self.outputs, list):
            return [tf.placeholder(
                dtype=output.dtype, shape=output.get_shape(),
                name=output.name.split(':')[0] + '/observed')
                    for output in self.outputs]
        return tf.placeholder(
            dtype=self.outputs.dtype, shape=self.outputs.get_shape(),
            name=self.outputs.name.split(':')[0] + 'observed/')

    def _process_loss(self, loss):
        if callable(loss):
            loss = loss(self.outputs, self.train_outputs)
        if not isinstance(loss, tf.Tensor):
            raise_error(ValueError, 'Unsupported loss type %s encountered.'
                        % type(loss))
        return loss

    def _process_optimizer(self, optimizer, optimizer_kwargs):
        if optimizer is None:
            return None
        if self.uses_external_optimizer:
            with tf.name_scope('external_optimizer'):
                if 'options' in optimizer_kwargs:
                    if 'disp' not in optimizer_kwargs['options']:
                        optimizer_kwargs['options']['disp'] = False
                else:
                    optimizer_kwargs['options'] = {'disp': False}
                return optimizer(self.loss, **optimizer_kwargs)
        if not isinstance(optimizer, tf.train.Optimizer):
            raise_error(ValueError, 'Unsupported optimizer type %s encountered.'
                        % type(optimizer))
        return optimizer

    def _train_op(self, train_op=None, optimizer=None, grads_processor=None):
        if train_op is not None:
            if callable(train_op):
                with self.graph.as_default():
                    return train_op()
            if not isinstance(train_op, tf.Operation):
                raise_error(ValueError, 'Unsupported train op type %s '
                                        'encountered.' % type(train_op))
            return train_op
        if grads_processor is not None:
            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(ys=self.loss, xs=trainable_vars)
            grads = grads_processor(grads)
            return optimizer.apply_gradients(
                grads_and_vars=zip(grads, trainable_vars),
                global_step=self.global_step)
        return optimizer.minimize(loss=self.loss, global_step=self.global_step)

    def get_feed_dict(self, data, is_train=False):
        if isinstance(data, np.ndarray):
            return {self.inputs: data}
        if isinstance(data, list) or isinstance(data, tuple):
            tensors = []
            if isinstance(self.inputs, list):
                tensors.extend(self.inputs)
            else:
                tensors.append(self.inputs)
            if is_train:
                if isinstance(self.train_outputs, list):
                    tensors.extend(self.train_outputs)
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
        # TODO: Allow the user to provide a map of names in the constructor.
        if isinstance(tensor, str):
            return tensor
        return tensor.name

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
                    loss=loss, optimizer=self._optimizer,
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
        raise_error(ValueError, 'Invalid op provided.')

    @staticmethod
    def _copy_ops_to_graph(ops, graph, variables=None, scope=''):
        if variables is None:
            variables = []
        if isinstance(ops, list):
            return [copy_op_to_graph(
                org_instance=op, to_graph=graph, variables=variables,
                copy_summaries=True, scope=scope) for op in ops]
        else:
            return copy_op_to_graph(
                org_instance=ops, to_graph=graph, variables=variables,
                copy_summaries=True, scope=scope)


class MultiLayerPerceptron(Model):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 softmax_output=True, use_log=True, train_outputs_one_hot=False,
                 loss=None, loss_summary=False, optimizer=None,
                 optimizer_opts=None, grads_processor=None):
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.use_log = use_log
        self.train_outputs_one_hot = train_outputs_one_hot
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        train_outputs = None if train_outputs_one_hot \
            else tf.placeholder(tf.int32, shape=[None])
        super(MultiLayerPerceptron, self).__init__(
            inputs=inputs, outputs=outputs, train_outputs=train_outputs,
            loss=loss, loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts, grads_processor=grads_processor)

    def __str__(self):
        return 'MultiLayerPerceptron[{}:{}:{}:{}]'.format(
            self.inputs.get_shape()[1], self.output_size,
            self.hidden_layer_sizes, self.softmax_output)

    def _output_op(self, inputs):
        hidden = inputs
        input_size = inputs.get_shape().dims[-1].value
        for layer_index, output_size in enumerate(self.hidden_layer_sizes):
            with tf.variable_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, output_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
                biases = tf.Variable(tf.zeros([output_size]), name='b')
                hidden = self.activation(tf.matmul(hidden, weights) + biases)
            input_size = output_size
        with tf.variable_scope('output_softmax_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, self.output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))), name='W')
            biases = tf.Variable(tf.zeros([self.output_size]), name='b')
            outputs = tf.matmul(hidden, weights) + biases
        if self.softmax_output and self.use_log:
            return tf.nn.log_softmax(outputs)
        elif self.softmax_output:
            return tf.nn.softmax(outputs)
        elif self.use_log:
            return tf.log(outputs)
        return outputs


class ADIOS(Model):
    """Architecture Deep In the Output Space.

    Composes an arbitrary input symbol with hierarchical multiple outputs.

    Arguments:
    ----------
        input_size : uint
        output_size : list
        hidden_layer_sizes : list
        activation : TF activation
        loss
        optimizer
        loss_summary
        grads_processor
    """
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 loss=None, loss_summary=False, optimizer=None,
                 optimizer_opts=None, grads_processor=None):
        assert len(output_size) == 2, "ADIOS works with exactly two outputs."
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        super(ADIOS, self).__init__(
            inputs=inputs, outputs=outputs, loss=loss,
            loss_summary=loss_summary, optimizer=optimizer,
            optimizer_opts=optimizer_opts, grads_processor=grads_processor)

    def _output_op(self, inputs):
        # Sanity check
        assert self.input_size == inputs.get_shape().dims[-1].value, \
            "Mismatch between the expected and actual input size."

        outputs = []

        # The first hidden and output layers are concatenated
        input_size = self.input_size
        output_size = self.output_size[0]
        with tf.variable_scope('output0_sigmoid_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            current = tf.sigmoid(tf.matmul(inputs, weights) + biases)
            outputs.append(current)
        if self.hidden_layer_sizes:
            hidden_size = self.hidden_layer_sizes[0]
            with tf.variable_scope('hidden0'):
                weights = tf.Variable(tf.random_normal(
                    [input_size, hidden_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([hidden_size]), name='b')
                hidden = self.activation(tf.matmul(inputs, weights) + biases)
                current = tf.concat(1, [current, hidden])

        # Add the rest of hidden layers
        input_size = current.get_shape().dims[-1].value
        for layer_index, hidden_size in enumerate(self.hidden_layer_sizes, 1):
            with tf.variable_scope('hidden' + str(layer_index)):
                weights = tf.Variable(tf.random_normal(
                    [input_size, hidden_size],
                    stddev=1.0 / np.math.sqrt(float(input_size))),
                    name='W'
                )
                biases = tf.Variable(tf.zeros([hidden_size]), name='b')
                current = self.activation(tf.matmul(current, weights) + biases)
                input_size = hidden_size

        # Add the final output layer
        output_size = self.output_size[1]
        with tf.variable_scope('output1_sigmoid_linear'):
            weights = tf.Variable(tf.random_normal(
                [input_size, output_size],
                stddev=1.0 / np.math.sqrt(float(input_size))),
                name='W'
            )
            biases = tf.Variable(tf.zeros([output_size]), name='b')
            current = tf.sigmoid(tf.matmul(current, weights) + biases)
            outputs.append(current)

        return outputs
