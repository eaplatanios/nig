import abc
import numpy as np
import tensorflow as tf
from six import with_metaclass

__author__ = 'eaplatanios'


def tf_aggregate_over_iterator(
        session, metric_tf_op, iterator, data_to_feed_dict,
        number_of_batches=-1, aggregating_function=np.mean):
    metrics = []
    if number_of_batches > -1:
        for batch_number in range(number_of_batches):
            data_batch = iterator.next()
            metrics.append(session.run(
                metric_tf_op, feed_dict=data_to_feed_dict(data_batch)))
    else:
        for data_batch in iterator:
            metrics.append(session.run(
                metric_tf_op, feed_dict=data_to_feed_dict(data_batch)))
    return aggregating_function(metrics)


class Metric(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def __str__(self):
        pass

    def __call__(self, prediction, truth, name='metric'):
        return self.tf_op(prediction, truth, name)

    @abc.abstractmethod
    def tf_op(self, prediction, truth, name='metric'):
        pass


class CrossEntropyOneHotEncodingMetric(Metric):
    """Note that this needs probability quantities as predictions (i.e., having
    gone through a softmax layer)."""
    def __str__(self):
        return 'cross_entropy'

    def tf_op(self, prediction, truth, name='cross_entropy'):
        with tf.name_scope('metric'):
            metric = -tf.reduce_sum(truth * prediction, reduction_indices=[1])
            metric = tf.reduce_mean(metric, name=name)
        return metric


class CrossEntropyIntegerEncodingMetric(Metric):
    """Note that this needs logit quantities as predictions (i.e., not having
    gone through a softmax layer)."""
    def __str__(self):
        return 'cross_entropy'

    def tf_op(self, prediction, truth, name='cross_entropy'):
        with tf.name_scope('metric'):
            metric = tf.nn.sparse_softmax_cross_entropy_with_logits(
                prediction, tf.to_int64(tf.squeeze(truth)))
            metric = tf.reduce_mean(metric, name=name)
        return metric


class AccuracyOneHotEncodingMetric(Metric):
    """Note that this needs probability quantities as predictions (i.e., having
    gone through a softmax layer)."""
    def __str__(self):
        return 'accuracy'

    def tf_op(self, prediction, truth, name='accuracy'):
        with tf.name_scope('metric'):
            metric = tf.equal(tf.argmax(prediction, 1), tf.argmax(truth, 1))
            metric = tf.reduce_mean(tf.cast(metric, tf.float32), name=name)
        return metric


class AccuracyIntegerEncodingMetric(Metric):
    """Note that this needs integer quantities as predictions (i.e., integer
    label corresponding to the most likely outcome)."""
    def __str__(self):
        return 'accuracy'

    def tf_op(self, prediction, truth, name='accuracy'):
        with tf.name_scope('metric'):
            metric = tf.nn.in_top_k(
                prediction, tf.to_int64(tf.squeeze(truth)), 1)
            metric = tf.reduce_mean(tf.cast(metric, tf.float32), name=name)
        return metric
