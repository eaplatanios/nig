import abc
import numpy as np
import tensorflow as tf
import os
import sys

from nig.data.iterators import NPArrayIterator
from nig.learning.optimizers import gradient_descent
from nig.utilities import logger

__author__ = 'eaplatanios'


def graph_context(func):
    def func_wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return func(self, *args, **kwargs)
    return func_wrapper


class Learner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, symbols, graph=tf.Graph(), session=None,
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64,
                 output_shape=None, predict_postprocess=lambda x: x):
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
        if output_shape is not None and isinstance(output_shape, list):
            self.output_shape = output_shape
        elif output_shape is not None:
            self.output_shape = [output_shape]
        with self.graph.as_default():
            self.inputs_op = tf.placeholder(inputs_dtype,
                                            [None] + self.input_shape)
            self.outputs_op = tf.placeholder(outputs_dtype,
                                             [None] + self.output_shape)
        self.predict_postprocess = predict_postprocess

    @graph_context
    def _init_session(self, option, saver, working_dir, checkpoint_file_prefix):
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
    def train(self, loss, train_data,
              optimizer=None,
              init_option=-1, callbacks=None, working_dir=os.getcwd(),
              checkpoint_file_prefix='checkpoint', restore_sequentially=False,
              save_trained=False):
        pass

    @abc.abstractmethod
    def _predict_op(self):
        pass

    @graph_context
    def predict(self, input_data, checkpoint=None, working_dir=os.getcwd(),
                checkpoint_file_prefix='checkpoint',
                restore_sequentially=False):
        if not isinstance(input_data, np.ndarray):
            iterator = self.predict_iterator(input_data, checkpoint)
            predictions = next(iterator)
            for batch in iterator:
                predictions = np.concatenate([predictions, batch], axis=0)
            return predictions
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(checkpoint, saver, working_dir,
                           checkpoint_file_prefix)
        predict_op = self.predict_postprocess(self._predict_op())
        return self.session.run(predict_op, {self.inputs_op: input_data})

    @graph_context
    def predict_iterator(self, input_data, checkpoint=None,
                         working_dir=os.getcwd(),
                         checkpoint_file_prefix='checkpoint',
                         restore_sequentially=False):
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(checkpoint, saver, working_dir,
                           checkpoint_file_prefix)
        predict_op = self.predict_postprocess(self._predict_op())
        for data_batch in input_data:
            yield self.session.run(predict_op, {self.inputs_op: data_batch})


class SimpleLearner(Learner):
    """Used for training a single TensorFlow model."""

    def __init__(self, symbol, graph=tf.Graph(), session=None,
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64,
                 output_shape=None, predict_postprocess=lambda x: x):
        super(SimpleLearner, self).__init__(symbol, graph, session,
                                            inputs_dtype, outputs_dtype,
                                            output_shape,
                                            predict_postprocess)
        with self.graph.as_default():
            self.predictions_op = self.symbols(self.inputs_op)

    def _train_op(self, loss, optimizer, loss_summary=False,
                  gradients_processor=None):
        global_step = tf.contrib.framework.get_or_create_global_step()
        if loss_summary:
            tf.scalar_summary(loss.op.name, loss)
        if gradients_processor is not None:
            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_variables)
            gradients = gradients_processor(gradients)
            train_op = optimizer.apply_gradients(zip(gradients,
                                                     trainable_variables),
                                                 global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    @graph_context
    def train(self, loss, train_data,
              optimizer=None,
              max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
              init_option=-1, callbacks=None, loss_summary=False,
              gradients_processor=None,
              run_metadata_collection_frequency=1000,
              trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
              checkpoint_file_prefix='checkpoint',
              restore_sequentially=False, save_trained=False):
        """

        Args:
            loss:
            train_data:
            optimizer:
            max_iter:
            loss_chg_tol:
            loss_chg_iter_below_tol:
            init_option:
            callbacks:
            loss_summary:
            gradients_processor:
            run_metadata_collection_frequency:
            trace_level (tf.RunOptions): Supported values include
                `tf.RunOptions.{NO_TRACE, SOFTWARE_TRACE HARDWARE_TRACE,
                FULL_TRACE}`.
            working_dir:
            checkpoint_file_prefix:
            restore_sequentially:
            save_trained:

        Returns:

        """
        if optimizer is None:
            gradient_descent(1e-1, decay_rate=0.99, learning_rate_summary=True)
        if isinstance(train_data, np.ndarray):
            train_data = NPArrayIterator(train_data, len(train_data),
                                         shuffle=False, cycle=False,
                                         keep_last=True)
        if callable(optimizer):
            optimizer = optimizer()
        loss_op = loss.tf_op(self.predictions_op, self.outputs_op)
        train_op = self._train_op(loss_op, optimizer, loss_summary,
                                  gradients_processor)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        train_data_iter = train_data.reset_copy(cycle=True)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(init_option, saver, working_dir,
                           checkpoint_file_prefix)
        for callback in callbacks:
            callback.initialize(self.graph, self.inputs_op,
                                self.outputs_op, self.predictions_op,
                                loss_op, summary_writer)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        for step in range(max_iter):
            data_batch = train_data_iter.next()
            feed_dict = {self.inputs_op: data_batch[0],
                         self.outputs_op: data_batch[1]}
            if run_metadata_collection_frequency > 0 \
                    and (step + 1) % run_metadata_collection_frequency == 0:
                run_options = tf.RunOptions(trace_level=trace_level)
                run_metadata = tf.RunMetadata()
                _, loss = self.session.run([train_op, loss_op],
                                           feed_dict,
                                           options=run_options,
                                           run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata,
                                                tag='step' + str(step),
                                                global_step=step)
            else:
                _, loss = self.session.run([train_op, loss_op], feed_dict)
            for callback in callbacks:
                callback(self.session, feed_dict, loss, step)
            if abs((prev_loss - loss) / prev_loss) < loss_chg_tol:
                iter_below_tol += 1
            else:
                iter_below_tol = 0
            if iter_below_tol >= loss_chg_iter_below_tol:
                logger.info('Loss value converged.')
                break
            prev_loss = loss
        if save_trained:
            Learner._save_checkpoint(self.session, saver, working_dir,
                                     checkpoint_file_prefix,
                                     max_iter)

    def _predict_op(self):
        return self.predictions_op


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
