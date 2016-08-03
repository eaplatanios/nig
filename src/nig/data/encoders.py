from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np

from nig.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


class Encoder(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, array):
        pass

    def encoder(self):
        return lambda array: self.encode(array)

    @abc.abstractmethod
    def decode(self, array):
        pass

    def decoder(self):
        return lambda array: self.decode(array)


class OneHotEncoder(Encoder):
    def __init__(self, encoding_size, encoding_map=None):
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
        if self.encoding_map is None:
            encoded_array[np.arange(number_of_instances), np.squeeze(array)] = 1
        else:
            encoded_array[np.arange(number_of_instances),
                          [self.encoding_map[label]
                           for label in np.squeeze(array)]] = 1
        return encoded_array

    def decode(self, array):
        if len(array.shape) != 2:
            error_message = 'The provided array has to be a matrix with the ' \
                            'rows indexing over the instances.'
            logger.error(error_message)
            raise ValueError(error_message)
        decoded_array = np.nonzero(array)[1]
        if self.encoding_map is None:
            return decoded_array
        if self.decoding_map is None:
            self.decoding_map = {code: label
                                 for label, code in self.encoding_map.items()}
        return np.vectorize(lambda code: self.decoding_map[code])(decoded_array)
