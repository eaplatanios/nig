import abc
import numpy as np
import tensorflow as tf
import os

from nig.data.iterators import NPArrayIterator
from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


class Learner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, symbols, graph=tf.Graph(), session=tf.Session(),
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64):
        self.symbols = symbols
        self.graph = graph
        self.session = session
        if isinstance(symbols, list):
            self.input_shape = self.symbols[0].input_shape
            self.output_shape = self.symbols[0].output_shape
            for symbol in symbols:
                if symbol.input_shape != self.input_shape:
                    error_message = 'All symbols input shapes must be equal.'
                    logger.error(error_message)
                    raise ValueError(error_message)
                if symbol.output_shape != self.output_shape:
                    error_message = 'All symbols output shapes must be equal.'
                    logger.error(error_message)
                    raise ValueError(error_message)
        else:
            self.input_shape = self.symbols.input_shape
            self.output_shape = self.symbols.output_shape
        with self.graph.as_default():
            self.inputs_op = tf.placeholder(inputs_dtype,
                                            [None] + self.input_shape)
            self.outputs_op = tf.placeholder(outputs_dtype,
                                             [None] + self.output_shape)
            # input_shape = [None] if self.input_shape is [1] \
            #     else [None] + self.input_shape
            # self.inputs_op = tf.placeholder(inputs_dtype, input_shape)
            # output_shape = [None] if self.output_shape is [1] \
            #     else [None] + self.output_shape
            # self.outputs_op = tf.placeholder(outputs_dtype, output_shape)

    def _initialize_session(self, option, saver, working_dir,
                            checkpoint_file_prefix):
        if option is None:
            option = False
        if isinstance(option, bool):
            if option:
                if self.session is None:
                    self.session = tf.Session()
                self.session.run(tf.initialize_all_variables())
                return
            elif self.session is None:
                raise ValueError('When the initialization option is a boolean '
                                 'value set to False, then a session needs to '
                                 'be provided.')
            else:
                return
        if saver is None:
            raise ValueError('When the initialization option is an integer, '
                             'indicating that a saved checkpoint should be '
                             'loaded, then a saver must also be provided.')
        if self.session is None:
            self.session = tf.Session()
        if isinstance(option, int):
            self._load_checkpoint(self.session, saver, working_dir,
                                  checkpoint_file_prefix, option)
        else:
            raise ValueError('Unsupported initialization.')
        return

    @staticmethod
    def _save_checkpoint(session, saver, working_dir, file_prefix, step):
        saver.save(session, os.path.join(working_dir, file_prefix),
                   global_step=step, latest_filename=file_prefix)

    @staticmethod
    def _load_checkpoint(session, saver, working_dir, file_prefix, step):
        if step > -1:
            checkpoint_file = os.path.join(working_dir, file_prefix) + str(step)
            if os.path.isfile(checkpoint_file):
                saver.restore(session, checkpoint_file)
            else:
                logger.warn('The requested checkpoint file does not exist. '
                            'All the variables are initialized to their '
                            'default values.')
                session.run(tf.initialize_all_variables())
        else:
            checkpoint_file = tf.train.latest_checkpoint(
                working_dir, latest_filename=file_prefix)
            saver.restore(session, checkpoint_file)

    @abc.abstractmethod
    def train(self, loss, train_data, learning_rate=1e-2, batch_size=-1,
              number_of_iterations=100000, initialization_option=-1,
              callbacks=None, working_dir=os.getcwd(),
              checkpoint_file_prefix='checkpoint', restore_sequentially=False,
              save_trained=False):
        pass

    @abc.abstractmethod
    def predict(self, input_data, checkpoint=None, working_dir=os.getcwd(),
                checkpoint_file_prefix='checkpoint',
                restore_sequentially=False):
        pass

    @abc.abstractmethod
    def predict_iterator(self, input_data, checkpoint=None,
                         working_dir=os.getcwd(),
                         checkpoint_file_prefix='checkpoint',
                         restore_sequentially=False):
        pass


class TensorFlowLearner(Learner):
    """Used for training a single TensorFlow model."""

    def __init__(self, symbol, graph=tf.Graph(), session=tf.Session(),
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64,
                 loss_summary=False, gradient_norm_summary=False):
        # TODO: We can train multiple symbols in parallel later on.
        super(TensorFlowLearner, self).__init__(symbol, graph, session,
                                                inputs_dtype, outputs_dtype)
        self.loss_summary = loss_summary
        self.gradient_norm_summary = gradient_norm_summary
        with self.graph.as_default():
            self.predictions_op = self.symbols.op(self.inputs_op)

    def _train_op(self, loss, optimizer):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.loss_summary:
            tf.scalar_summary(loss.op.name, loss)
        if self.gradient_norm_summary:
            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_variables)
            gradients_norm = tf.reduce_sum([tf.nn.l2_loss(gradient)
                                            for gradient in gradients],
                                           name='gradients_norm')
            tf.scalar_summary(gradients_norm.op.name, gradients_norm)
            train_op = optimizer.apply_gradients(zip(gradients,
                                                     trainable_variables),
                                                 global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def _data_to_feed_dict(self, data_batch):
        return {self.inputs_op: data_batch[0], self.outputs_op: data_batch[1]}

    def train(self, loss, train_data, learning_rate=1e-2, batch_size=-1,
              number_of_iterations=100000, initialization_option=-1,
              callbacks=None, working_dir=os.getcwd(),
              checkpoint_file_prefix='checkpoint', restore_sequentially=False,
              save_trained=False):
        if isinstance(train_data, np.ndarray):
            train_data = NPArrayIterator(train_data, len(train_data),
                                         shuffle=False, cycle=False,
                                         keep_last_batch=True)
        with self.graph.as_default():
            loss_op = loss.tf_op(self.predictions_op, self.outputs_op)
            # TODO: Provide optimizer as input argument.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = self._train_op(loss_op, optimizer)
            summary_op = tf.merge_all_summaries()
            train_data_iter = train_data.reset_copy(
                batch_size=batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True
            ) if batch_size > -1 else train_data.reset_copy(
                shuffle=True, cycle=True, cycle_shuffle=True
            )
            saver = tf.train.Saver(restore_sequentially=restore_sequentially)
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())
            # self._initialize_session(initialization_option, saver, working_dir,
            #                          checkpoint_file_prefix)
            for callback in callbacks:
                callback.initialize(self.graph, self.inputs_op,
                                    self.outputs_op, self.predictions_op,
                                    loss_op, summary_op)
            for step in range(number_of_iterations):
                train_data_batch = train_data_iter.next()
                feed_dict = self._data_to_feed_dict(train_data_batch)
                _, loss = self.session.run([train_op, loss_op],
                                           feed_dict=feed_dict)
                for callback in callbacks:
                    callback.call(self.session, feed_dict, loss, step)
            # TD: Add option to skip saving a checkpoint here.
            # Learner._save_checkpoint(session, saver, self.working_dir,
            #                          self.checkpoint_file_prefix,
            #                          number_of_iterations)

    def predict(self, input_data, checkpoint=None, working_dir=os.getcwd(),
                checkpoint_file_prefix='checkpoint',
                restore_sequentially=False):
        if isinstance(input_data, np.ndarray):
            with self.graph.as_default():
                saver = tf.train.Saver(restore_sequentially=restore_sequentially)
                self._initialize_session(checkpoint, saver, working_dir,
                                         checkpoint_file_prefix)
                return self.session.run([self.predictions_op],
                                        feed_dict={self.inputs_op: input_data})
        else:
            predictions = np.empty(0)
            for predictions_batch in self.predict_iterator(input_data,
                                                           checkpoint):
                predictions = np.hstack([predictions, predictions_batch])
            return predictions

    def predict_iterator(self, input_data, checkpoint=None,
                         working_dir=os.getcwd(),
                         checkpoint_file_prefix='checkpoint',
                         restore_sequentially=False):
        with self.graph.as_default():
            saver = tf.train.Saver(restore_sequentially=restore_sequentially)
            self._initialize_session(checkpoint, saver, working_dir,
                                     checkpoint_file_prefix)
            for data_batch in input_data:
                yield self.session.run([self.predictions_op],
                                       feed_dict={self.inputs_op: data_batch})


# class TensorFlowMultiModelValidationSetLearner(Learner):
#     """Used for training multiple TensorFlow models that have the same input
#     and
#     predict the same quantities, using a validation data set to pick the best
#     model."""
#     def train(self, models, train_data, eval_data=None, test_data=None):
#         pass
#
#     def predict(self, input_data, checkpoint=-1, session=None):
#         pass
#
#     def predict_iterator(self, input_data):
#         pass
#
#
# class TensorFlowMultiModelCrossValidationLearner(Learner):
#     """Used for training multiple TensorFlow models that have the same input
#     and
#     predict the same quantities, using cross-validation to pick the best
#     model."""
#     def train(self, models, train_data, eval_data=None, test_data=None):
#         pass
#
#     def predict(self, input_data, checkpoint=-1, session=None):
#         pass
#
#     def predict_iterator(self, input_data):
#         pass
#
#
# class TensorFlowMultiModelNIGLearner(Learner):
#     """Used for training multiple TensorFlow models that have the same input
#     and
#     predict the same quantities, using the NIG agreement-driven approach to
#     train them jointly and allow them to make predictions jointly."""
#     def train(self, models, train_data, eval_data=None, test_data=None):
#         pass
#
#     def predict(self, input_data, checkpoint=-1, session=None):
#         pass
#
#     def predict_iterator(self, input_data):
#         pass
