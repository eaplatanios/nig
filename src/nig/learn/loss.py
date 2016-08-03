import abc
import tensorflow as tf

__author__ = 'Emmanouil Antonios Platanios'


class Loss(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def function(self, prediction, truth):
        pass


class CrossEntropyOneHotEncodingLoss(Loss):
    """Note that this needs probability quantities as predictions (i.e., having
    gone through a softmax layer)."""
    def function(self, prediction, truth):
        cross_entropy = -tf.reduce_sum(truth * prediction,
                                       reduction_indices=[1])
        return tf.reduce_mean(cross_entropy, name='cross_entropy_mean')


class CrossEntropyIntegerEncodingLoss(Loss):
    """Note that this needs logit quantities as predictions (i.e., not having
    gone through a softmax layer)."""
    def function(self, prediction, truth):
        truth = tf.to_int64(truth)
        cross_entropy = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, truth)
        return tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
