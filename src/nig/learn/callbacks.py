import abc
import numpy as np
import sys
import tensorflow as tf
import os

from nig.learn.metrics import tf_aggregate_over_iterator
from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'

__NOT_INITIALIZED_ERROR__ = 'The callback has not been initialized yet.'


class Callback(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, frequency=1):
        self.frequency = frequency

    @abc.abstractmethod
    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_op):
        pass

    def __call__(self, session, feed_dict=None, loss=None, global_step=None):
        if (global_step + 1) % self.frequency == 0 or global_step == 0:
            self._execute(session, feed_dict, loss, global_step)

    @abc.abstractmethod
    def _execute(self, session, feed_dict, loss, global_step):
        pass


class LoggerCallback(Callback):
    def __init__(self, name='Logger Callback',
                 log_format='{:>20} - | {:>10d} | {:>10.4e} |',
                 header='{:>20} - | {:>10} | {:>10} |'.format('Logger Callback',
                                                              'Step', 'Loss'),
                 header_frequency=sys.maxsize, frequency=1):
        super(LoggerCallback, self).__init__(frequency)
        self.name = name
        self.log_format = log_format
        self.header = header
        self.header_frequency = header_frequency

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_op):
        pass

    def _execute(self, session, feed_dict, loss, global_step):
        if global_step % self.header_frequency == 0:
            logger.info(self.header)
        logger.info(self.log_format.format(self.name, global_step+1, loss))


class SummaryWriterCallback(Callback):
    def __init__(self, working_dir=os.getcwd(), frequency=100):
        super(SummaryWriterCallback, self).__init__(frequency)
        self.working_dir = working_dir
        self.summary_op = None
        self.summary_writer = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_op):
        self.summary_op = summary_op
        self.summary_writer = tf.train.SummaryWriter(self.working_dir, graph)

    def _execute(self, session, feed_dict, loss, global_step):
        if self.summary_op is None:
            raise ValueError(__NOT_INITIALIZED_ERROR__)
        summary = session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()


class CheckpointWriterCallback(Callback):
    def __init__(self, variable_list=None, max_to_keep=5,
                 keep_checkpoint_every_n_hours=10000.0, working_dir=os.getcwd(),
                 file_prefix='checkpoint',
                 frequency=1000):
        super(CheckpointWriterCallback, self).__init__(frequency)
        self.variable_list = variable_list
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.working_dir = working_dir
        self.file_prefix = file_prefix
        self.saver = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_op):
        self.saver = tf.train.Saver(
            var_list=self.variable_list, max_to_keep=self.max_to_keep,
            keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

    def _execute(self, session, feed_dict=None, loss=None, global_step=None):
        if self.saver is None:
            raise ValueError(__NOT_INITIALIZED_ERROR__)
        self.saver.save(session, os.path.join(self.working_dir,
                                              self.file_prefix),
                        global_step=global_step,
                        latest_filename=self.file_prefix)


class EvaluationCallback(Callback):
    def __init__(self, iterator, metrics, number_of_batches=-1,
                 aggregating_function=np.mean, name='Evaluation Callback',
                 log_format=None, header=None, header_frequency=sys.maxsize,
                 frequency=1):
        super(EvaluationCallback, self).__init__(frequency)
        self.iterator = iterator
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.number_of_batches = number_of_batches
        self.aggregating_function = aggregating_function
        self.name = name
        self.log_format = log_format if log_format is not None \
            else '{:>20} - | {:>10d} | {:>10.4e} |' * len(self.metrics)
        self.header = header if header is not None \
            else ('{:>20} - | {:>10} | {:>10} |' * len(self.metrics)) \
            .format(name, 'Step', *[str(metric)
                                    for metric in self.metrics])
        self.header_frequency = header_frequency
        self.inputs_op = None
        self.outputs_op = None
        self.eval_ops = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_op):
        self.inputs_op = inputs_op
        self.outputs_op = outputs_op
        self.eval_ops = [metric.tf_op(predictions_op, outputs_op)
                         for metric in self.metrics]

    def _execute(self, session, feed_dict=None, loss=None, global_step=None):
        self.iterator.reset()
        metrics = []
        for eval_op in self.eval_ops:
            metrics.append(tf_aggregate_over_iterator(
                session, eval_op, self.iterator,
                lambda data_batch: {self.inputs_op: data_batch[0],
                                    self.outputs_op: data_batch[1]},
                self.number_of_batches, self.aggregating_function))
        if global_step % self.header_frequency == 0:
            logger.info(self.header)
        logger.info(self.log_format.format(self.name, global_step+1, *metrics))
