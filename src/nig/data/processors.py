from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np
import tensorflow as tf
from enum import Enum
from six import with_metaclass

from nig.functions import PipelineFunction

__author__ = 'eaplatanios'


class Type(Enum):
    np = 1
    numpy = 1
    pd = 2
    pandas = 2
    tf = 3
    tensorflow = 3

    def process_function(self, processor):
        if self.value == 1:
            return processor.process_np
        elif self.value == 2:
            return processor.process_pd
        elif self.value == 3:
            return processor.process_tf

    def process_pipeline_function(self, processor):
        if self.value == 1:
            return PipelineFunction(processor.process_np)
        elif self.value == 2:
            return PipelineFunction(processor.process_pd)
        elif self.value == 3:
            return PipelineFunction(processor.process_tf)  # TODO: Maybe Symbol?


class Processor(with_metaclass(abc.ABCMeta, PipelineFunction)):
    """A :class:`Processor` receives some data as input and provides a function
    for transforming that data to a new representation. Depending on what
    each processor supports, that data can be in the form of a
    :class:`np.ndarray`, a :class:`pd.Series`, a :class:`pd.DataFrame`, or a
    :class:`tf.Tensor`.

    Note:
        Processors are not supposed to aggregate information over samples, but
        to rather process the information for each data sample separately. It is
        thus required for processor preserve the sample information. For
        numpy arrays and tensorflow tensors that corresponds to not altering
        the size of the first dimension, and for pandas data structures that
        corresponds to not alterning the index.
    """
    def __init__(self, processor_type=Type.np):
        super(Processor, self).__init__(processor_type.process_function(self))

    @abc.abstractmethod
    def process_np(self, array):
        pass

    @abc.abstractmethod
    def process_pd(self, data):
        pass

    @abc.abstractmethod
    def process_tf(self, tensor):
        # TODO: Should figure out what this returns to make it pipelinable.
        pass


class ColumnsExtractor(Processor):
    """Extracts the specified columns from the input array, forming a new
    array."""
    def __init__(self, columns, processor_type=Type.np):
        super(ColumnsExtractor, self).__init__(processor_type)
        self.columns = columns if isinstance(columns, list) else [columns]

    def process_np(self, array):
        return array[:, self.columns]

    def process_pd(self, data):
        return data[self.columns]

    def process_tf(self, tensor):
        raise NotImplementedError()


class DataTypeEncoder(Processor):
    def __init__(self, dtype, processor_type=Type.np):
        super(Processor, self).__init__(processor_type)
        self.dtype = dtype

    def process_np(self, array):
        return array.astype(self.dtype)

    def process_pd(self, data):
        raise NotImplementedError()

    def process_tf(self, tensor):
        return tf.cast(tensor, dtype=self.dtype)


class ReshapeEncoder(Processor):
    def __init__(self, shape, processor_type=Type.np):
        super(ReshapeEncoder, self).__init__(processor_type)
        self.shape = shape

    def process_np(self, array):
        return array.reshape(self.shape)

    def process_pd(self, data):
        raise NotImplementedError()

    def process_tf(self, tensor):
        return tf.reshape(tensor, shape=self.shape)


class OneHotEncoder(Processor):
    def __init__(self, encoding_size, encoding_map=None,
                 processor_type=Type.np):
        super(OneHotEncoder, self).__init__(processor_type)
        self.encoding_size = encoding_size
        self.encoding_map = encoding_map

    def process_np(self, array):
        num_samples = array.shape[0]
        encoded_array = np.zeros([num_samples, self.encoding_size])
        index_offset = np.arange(num_samples) * self.encoding_size
        if self.encoding_map is None:
            encoded_array.flat[index_offset + array.ravel()] = 1
            return encoded_array
        map_values = np.vectorize(lambda k: self.encoding_map[k])
        encoded_array.flat[index_offset + map_values(array.ravel())] = 1
        return encoded_array

    def process_pd(self, data):
        raise NotImplementedError()

    def process_tf(self, tensor):
        if self.encoding_map is None:
            return tf.one_hot(tensor, self.encoding_size)
        raise NotImplementedError()


class OneHotDecoder(Processor):
    def __init__(self, encoding_size, encoding_map=None, decoding_map=None,
                 processor_type=Type.np):
        super(OneHotDecoder, self).__init__(processor_type)
        self.encoding_size = encoding_size
        if encoding_map is None:
            self.decoding_map = decoding_map
        else:
            self.decoding_map = {v: k for k, v in encoding_map.items()}

    def process_np(self, array):
        decoded_array = array.nonzero()[1]
        if self.decoding_map is None:
            return decoded_array
        map_values = np.vectorize(lambda v: self.decoding_map[v])
        return map_values(decoded_array)

    def process_pd(self, data):
        raise NotImplementedError()

    def process_tf(self, tensor):
        raise NotImplementedError()
