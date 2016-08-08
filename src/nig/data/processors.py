from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np
import tensorflow as tf

from six import with_metaclass

from nig.functions import PipelineFunction

__author__ = 'eaplatanios'


class Processor(with_metaclass(abc.ABCMeta, PipelineFunction)):
    """A :class:`Processor` receives some data as input and provides a function
    for transforming that data to a new representation.

    Note:
        Processors are not supposed to aggregate information over samples, but
        to rather process the information for each data sample separately. It is
        thus required for processor preserve the sample information (such as
        a pandas time index, for example).
    """
    def __init__(self):
        super(Processor, self).__init__(self.process)

    @abc.abstractmethod
    def process(self, data):
        pass


class NPProcessor(with_metaclass(abc.ABCMeta, Processor)):
    """A :class:`NPProcessor` receives a :class:`np.ndarray` as input and
    provides a function for transforming it to a new representation.

    Note:
        The first dimension of the input array is always preserved as it is
        meant to correspond to an index over samples. Processors are required
        not to alter that index.
    """
    def __init__(self):
        super(NPProcessor, self).__init__()

    @abc.abstractmethod
    def process(self, data):
        pass


class TFProcessorMixin(with_metaclass(abc.ABCMeta)):
    """A :class:`TFProcessorMixin` is supposed to be used mixed in  with the
    :class:`NPProcessor` abstract class, whenever a TensorFlow op can be defined
    for the corresponding transform, operating on a :class:`tf.Tensor`. This
    allows for the corresponding transform to be computed by TensorFlow.

    Note:
        The first dimension of the input tensor is always preserved as it is
        meant to correspond to an index over samples. Processors are required
        not to alter that index.
    """
    @abc.abstractmethod
    def process_op(self, tensor):  # TODO: Somehow make this pipelinable.
        pass


class NPColumnsExtractor(NPProcessor):
    """Extracts the specified columns from the input array, forming a new
    array."""
    def __init__(self, columns):
        super(NPColumnsExtractor, self).__init__()
        self.columns = columns if isinstance(columns, list) else [columns]

    def process(self, data):
        return data[:, self.columns]

    # TODO: Add TensorFlow mixin support.


class NPTFDataTypeEncoder(NPProcessor, TFProcessorMixin):
    def __init__(self, dtype):
        super(NPTFDataTypeEncoder, self).__init__()
        self.dtype = dtype

    def process(self, data):
        return data.astype(self.dtype)

    def process_op(self, tensor):
        return tf.cast(tensor, dtype=self.dtype)


class NPTFReshapeEncoder(NPProcessor, TFProcessorMixin):
    def __init__(self, shape):
        super(NPTFReshapeEncoder, self).__init__()
        self.shape = shape

    def process(self, data):
        return data.reshape(self.shape)

    def process_op(self, tensor):
        return tf.reshape(tensor, shape=self.shape)


class NPOneHotEncoder(NPProcessor, TFProcessorMixin):
    def __init__(self, encoding_size):
        super(NPOneHotEncoder, self).__init__()
        self.encoding_size = encoding_size

    def process(self, data):
        num_samples = data.shape[0]
        encoded_array = np.zeros([num_samples, self.encoding_size])
        index_offset = np.arange(num_samples) * self.encoding_size
        encoded_array.flat[index_offset + data.ravel()] = 1
        return encoded_array

    def process_op(self, tensor):
        return tf.one_hot(tensor, self.encoding_size)


class PDProcessor(with_metaclass(abc.ABCMeta, Processor)):
    """A :class:`PDProcessor` receives a :class:`pd.Series` or a
    :class:`pd.DataFrame` as input and provides a function for transforming
    it to a new representation.

    Note:
        The index of the input pandas series or data frame is always preserved
        as it is meant to correspond to an index over samples. Processors are
        required not to alter that index.
    """
    def __init__(self):
        super(PDProcessor, self).__init__()

    @abc.abstractmethod
    def process(self, data):
        pass
