from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np

from nig.functions import pipeline
from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


class Encoder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.encoder = Encoder.__encoder_function(self)
        self.decoder = Encoder.__decoder_function(self)

    @abc.abstractmethod
    def encode(self, array):
        pass

    @abc.abstractmethod
    def decode(self, array):
        pass

    @staticmethod
    @pipeline(min_num_args=1)
    def __encoder_function(encoder, data):
        return encoder.encode(data)

    @staticmethod
    @pipeline(min_num_args=1)
    def __decoder_function(encoder, data):
        return encoder.decode(data)


class DataTypeEncoder(Encoder):
    def __init__(self, target_dtype, source_dtype=None):
        super(DataTypeEncoder, self).__init__()
        self.target_dtype = target_dtype
        self.source_dtype = source_dtype

    def encode(self, array):
        self.source_dtype = array.dtype
        return array.astype(self.target_dtype)

    def decode(self, array):
        if self.source_dtype is None:
            raise ValueError('No source data type provided nor an array to '
                             'encode before decoding, and thus inferring the '
                             'data type.')
        return array.astype(self.target_dtype)


class ReshapeEncoder(Encoder):
    def __init__(self, target_shape, source_shape=None):
        super(ReshapeEncoder, self).__init__()
        self.target_shape = target_shape
        self.source_shape = source_shape

    def encode(self, array):
        self.source_shape = array.shape
        return array.reshape(self.target_shape)

    def decode(self, array):
        if self.source_shape is None:
            raise ValueError('No source shape provided nor an array to encode '
                             'before decoding, and thus inferring the shape.')
        return array.reshape(self.target_shape)


class OneHotEncoder(Encoder):
    def __init__(self, encoding_size, encoding_map=None):
        super(OneHotEncoder, self).__init__()
        self.encoding_size = encoding_size
        self.encoding_map = encoding_map
        self.decoding_map = None

    def encode(self, array):
        if len(array.shape) > 2 or (len(array.shape) == 2
                                    and array.shape[0] != 1
                                    and array.shape[1] != 1):
            error_message = 'The provided array has to be either a column ' \
                            'vector (i.e., rows index over instances).'
            logger.error(error_message)
            raise ValueError(error_message)
        number_of_instances = array.shape[0]
        encoded_array = np.zeros([number_of_instances, self.encoding_size])
        index_offset = np.arange(number_of_instances) * self.encoding_size
        if self.encoding_map is None:
            encoded_array.flat[index_offset + array.ravel()] = 1
        else:
            map_values = np.vectorize(lambda k: self.encoding_map[k])
            encoded_array.flat[index_offset + map_values(array.ravel())] = 1
        return encoded_array

    def decode(self, array):
        if len(array.shape) != 2:
            error_message = 'The provided array has to be a matrix with the ' \
                            'rows indexing over the instances.'
            logger.error(error_message)
            raise ValueError(error_message)
        decoded_array = array.nonzero()[1]
        if self.encoding_map is None:
            return decoded_array
        if self.decoding_map is None:
            self.decoding_map = {v: k for k, v in self.encoding_map.items()}
        map_values = np.vectorize(lambda v: self.decoding_map[v])
        return map_values(decoded_array)
