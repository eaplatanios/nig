import abc
import numpy as np
import sys
import tensorflow as tf
import os

from nig.learning.metrics import tf_aggregate_over_iterator
from nig.utilities import logger

__author__ = 'eaplatanios'

__NOT_INITIALIZED_ERROR__ = 'The callback has not been initialized yet.'


class Callback(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, frequency=1):
        self.frequency = frequency

    @abc.abstractmethod
    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
        pass

    def __call__(self, session, feed_dict=None, loss=None, global_step=None):
        if (global_step + 1) % self.frequency == 0 or global_step == 0:
            self._execute(session, feed_dict, loss, global_step)

    @abc.abstractmethod
    def _execute(self, session, feed_dict, loss, global_step):
        pass


class LoggerCallback(Callback):
    def __init__(self, frequency=100, name='logger_callback',
                 log_format='{:>20} - | {:>10d} | {:>10.4e} |',
                 header='{:>20} - | {:>10} | {:>10} |'
                        .format('Logger Callback', 'Step', 'Loss'),
                 header_frequency=sys.maxsize):
        super(LoggerCallback, self).__init__(frequency)
        self.name = name
        self.log_format = log_format
        self.header = header
        self.header_frequency = header_frequency

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
        pass

    def _execute(self, session, feed_dict, loss, global_step):
        if global_step % self.header_frequency == 0:
            logger.info(self.header)
        logger.info(self.log_format.format(self.name, global_step+1, loss))


class SummaryWriterCallback(Callback):
    def __init__(self, frequency=100, working_dir=os.getcwd()):
        super(SummaryWriterCallback, self).__init__(frequency)
        self.working_dir = working_dir
        self.summary_op = None
        self.summary_writer = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = summary_writer

    def _execute(self, session, feed_dict, loss, global_step):
        if self.summary_op is None:
            raise ValueError(__NOT_INITIALIZED_ERROR__)
        summary = session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()


class VariableStatisticsSummaryWriterCallback(Callback):
    def __init__(self, frequency=100, variables='trainable', statistics=None,
                 histogram=True, name='trainable',
                 working_dir=os.getcwd()):
        super(VariableStatisticsSummaryWriterCallback, self).__init__(frequency)
        self.variables = variables
        if statistics is None:
            statistics = {'mean': tf.reduce_mean,
                          'std_dev': self._std_dev,
                          'max': tf.reduce_max,
                          'min': tf.reduce_min}
        self.statistics = statistics
        self.histogram = histogram
        self.name = name
        self.working_dir = working_dir
        self.summary_op = None
        self.summary_writer = None

    @staticmethod
    def _std_dev(variable):
        return tf.sqrt(tf.reduce_sum(tf.square(variable -
                                               tf.reduce_mean(variable))))

    def _variable_statistics_summaries(self):
        summaries = []
        with tf.name_scope('summaries'):
            for variable in self.variables:
                for name, statistic in self.statistics.items():
                    summaries.append(tf.scalar_summary('variables/' + name +
                                                       '/' + variable.name,
                                                       statistic(variable)))
                if self.histogram:
                    summaries.append(tf.histogram_summary('variables/' +
                                                          variable.name,
                                                          variable))
            return tf.merge_summary(summaries, name='variables' + self.name)

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
        if self.variables is 'trainable':
            self.variables = tf.trainable_variables()
        elif self.variables is 'all':
            self.variables = tf.all_variables()
        elif isinstance(self.variables, str):
            self.variables = [v for v in tf.all_variables()
                              if v.name == self.variables][0]
        elif isinstance(self.variables, tf.Variable):
            self.variables = [self.variables]
        elif isinstance(self.variables, list):
            self.variables = [v for v in tf.all_variables()
                              if v.name in self.variables]
        self.summary_op = self._variable_statistics_summaries()
        self.summary_writer = summary_writer

    def _execute(self, session, feed_dict, loss, global_step):
        if self.summary_op is None:
            raise ValueError(__NOT_INITIALIZED_ERROR__)
        summary = session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()


class CheckpointWriterCallback(Callback):
    def __init__(self, frequency=1000, variable_list=None, max_to_keep=5,
                 keep_checkpoint_every_n_hours=10000.0, working_dir=os.getcwd(),
                 file_prefix='checkpoint'):
        super(CheckpointWriterCallback, self).__init__(frequency)
        self.variable_list = variable_list
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.working_dir = working_dir
        self.file_prefix = file_prefix
        self.saver = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
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
    def __init__(self, frequency, iterator, metrics, number_of_batches=-1,
                 aggregating_function=np.mean, name='eval_callback',
                 log_format=None, header=None, header_frequency=sys.maxsize,
                 summary=True, working_dir=os.getcwd()):
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
        self.summary = summary
        self.working_dir = working_dir
        self.summary_writer = None
        self.inputs_op = None
        self.outputs_op = None
        self.eval_ops = None

    def initialize(self, graph, inputs_op, outputs_op, predictions_op, loss_op,
                   summary_writer):
        self.inputs_op = inputs_op
        self.outputs_op = outputs_op
        self.eval_ops = [metric.tf_op(predictions_op, outputs_op)
                         for metric in self.metrics]
        if self.summary:
            self.summary_writer = summary_writer

    def _execute(self, session, feed_dict=None, loss=None, global_step=None):
        self.iterator.reset()
        metrics = []
        for index, eval_op in enumerate(self.eval_ops):
            value = tf_aggregate_over_iterator(
                session, eval_op, self.iterator,
                lambda data_batch: {self.inputs_op: data_batch[0],
                                    self.outputs_op: data_batch[1]},
                self.number_of_batches, self.aggregating_function)
            metrics.append(value)
            if self.summary:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.tag = self.name + '/' + str(self.metrics[index])
                summary_value.simple_value = float(value)
                self.summary_writer.add_summary(summary, global_step)
        if global_step % self.header_frequency == 0:
            logger.info(self.header)
        logger.info(self.log_format.format(self.name, global_step+1, *metrics))
        if self.summary:
            self.summary_writer.flush()
