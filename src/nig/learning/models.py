import abc
import itertools
import numpy as np
import tensorflow as tf
from six import with_metaclass

from nig.utilities.generic import logger

__author__ = 'eaplatanios'


class Model(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, inputs, outputs, loss=None, optimizer=None,
                 loss_summary=False, grads_processor=None):
        if isinstance(inputs, list):
            self.graph = inputs[0].graph
        else:
            self.graph = inputs.graph
        self.inputs = inputs
        self.outputs = outputs
        if isinstance(outputs, list) and isinstance(loss, list):
            if len(outputs) != len(loss):
                raise ValueError('The number of provided output ops must '
                                 'match the number of provided loss ops.')
        self.loss = loss
        self.trainable = loss is not None and optimizer is not None
        if self.trainable:
            with self.graph.as_default():
                self.optimizer = optimizer
                self.tf_optimizer = self._process_optimizer(optimizer)
                if isinstance(outputs, list):
                    self.train_outputs = [tf.placeholder(
                        dtype=output.dtype, shape=output.get_shape(),
                        name=output.name.split(':')[0] + '/observed')
                                          for output in outputs]
                else:
                    self.train_outputs = tf.placeholder(
                        dtype=outputs.dtype, shape=outputs.get_shape(),
                        name=outputs.name.split(':')[0] + 'observed/')
                self.loss_op = loss.tf_op(outputs, self.train_outputs)
                self.train_op = self._train_op(loss_summary, grads_processor)

    @staticmethod
    def _process_optimizer(optimizer):
        if callable(optimizer):
            optimizer = optimizer()
        if not isinstance(optimizer, tf.train.Optimizer):
            raise ValueError('Unsupported optimizer encountered.')
        return optimizer

    def _train_op(self, loss_summary=False, grads_processor=None):
        global_step = tf.contrib.framework.get_or_create_global_step()
        if loss_summary:
            tf.scalar_summary(self.loss_op.op.name, self.loss_op)
        if grads_processor is not None:
            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(ys=self.loss_op, xs=trainable_vars)
            grads = grads_processor(grads)
            return self.tf_optimizer.apply_gradients(
                grads_and_vars=zip(grads, trainable_vars),
                global_step=global_step)
        return self.tf_optimizer.minimize(
            loss=self.loss_op, global_step=global_step)

    def get_feed_dict(self, input_data_batch, output_data_batch=None):
        if isinstance(input_data_batch, dict):
            feed_dict = {k.name: v for k, v in input_data_batch.items()}
        elif isinstance(input_data_batch, list) \
                or isinstance(input_data_batch, tuple):
            feed_dict = dict(zip(self.inputs, input_data_batch))
        else:
            feed_dict = {self.inputs: input_data_batch}
        if output_data_batch is None:
            return feed_dict
        if isinstance(output_data_batch, dict):
            feed_dict.update({k.name + '/observed': v
                              for k, v in output_data_batch.items()})
        elif isinstance(input_data_batch, list) \
                or isinstance(input_data_batch, tuple):
            feed_dict.update(dict(zip(self.train_outputs, output_data_batch)))
        else:
            feed_dict[self.train_outputs] = output_data_batch
        return feed_dict

    def copy_to_graph(self, graph, scope=''):
        for variable in self._variables():
            tf.contrib.copy_graph.copy_variable_to_graph(
                org_instance=variable, to_graph=graph, scope=scope)
        inputs = self._copy_ops_to_graph(self.inputs, graph, scope)
        outputs = self._copy_ops_to_graph(self.outputs, graph, scope)
        return Model(inputs, outputs, self.loss, self.optimizer)

    def _variables(self):
        all_variables = {var.name.split(':')[0]: var
                         for var in tf.all_variables()}
        if isinstance(self.outputs, list):
            return set(all_variables[var]
                       for var in itertools.chain.from_iterable(
                self._op_variables(output) for output in self.outputs))
        return set(all_variables[var]
                   for var in self._op_variables(self.outputs))

    def _op_variables(self, op):
        if isinstance(op, tf.Operation):
            return set(itertools.chain(
                *[self._op_variables(input_op) for input_op in op.inputs]))
        elif isinstance(op, tf.Tensor):
            variables = set()
            if op.op.type == 'Variable':
                variables.add(op.op.name)
            if len(op.op.inputs) > 0:
                for input_op in op.op.inputs:
                    variables.update(self._op_variables(input_op))
            return variables
        logger.error('Invalid op provided.')
        raise ValueError('Invalid op provided.')

    @staticmethod
    def _copy_ops_to_graph(ops, graph, scope=''):
        if isinstance(ops, list):
            return [tf.contrib.copy_graph.copy_op_to_graph(
                org_instance=op, to_graph=graph, variables=[], scope=scope)
                    for op in ops]
        else:
            return tf.contrib.copy_graph.copy_op_to_graph(
                org_instance=ops, to_graph=graph, variables=[], scope=scope)


class MultiLayerPerceptron(Model):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation,
                 softmax_output=True, use_log=True, loss=None, optimizer=None,
                 loss_summary=False, grads_processor=None):
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.softmax_output = softmax_output
        self.use_log = use_log
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        super(MultiLayerPerceptron, self).__init__(
            inputs=inputs, outputs=outputs, loss=loss, optimizer=optimizer,
            loss_summary=loss_summary, grads_processor=grads_processor)

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
                 loss=None, optimizer=None,
                 loss_summary=False, grads_processor=None):
        assert len(output_size) == 2, "ADIOS works with exactly two outputs."
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        inputs = tf.placeholder(tf.float32, shape=[None, input_size])
        outputs = self._output_op(inputs)
        super(ADIOS, self).__init__(
            inputs=inputs, outputs=outputs, loss=loss, optimizer=optimizer,
            loss_summary=loss_summary, grads_processor=grads_processor)

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
