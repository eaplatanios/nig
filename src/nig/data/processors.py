# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import abc
import numpy as np
import pandas as pd

from six import with_metaclass

from nig.utilities.functions import PipelineFunction

__author__ = 'eaplatanios'

__all__ = ['Processor', 'ColumnsExtractor', 'DataTypeEncoder', 'ReshapeEncoder',
           'OneHotEncoder', 'OneHotDecoder']

__UNSUPPORTED_DATA_TYPE_ERROR__ = 'Unsupported data type %s for processor.'


class Processor(with_metaclass(abc.ABCMeta, PipelineFunction)):
    """A :class:`Processor` receives some data as input and provides a function
    for transforming that data to a new representation. Depending on what
    each processor supports, that data can be in various forms, including
    :class:`np.ndarray`, :class:`pd.Series`, and :class:`pd.DataFrame`.

    Note:
        Processors are not supposed to aggregate information over samples, but
        to rather process the information for each data sample separately. It is
        thus required for processors to preserve the sample information. For
        numpy arrays and tensorflow tensors that corresponds to not altering
        the size of the first dimension, and for pandas data structures that
        corresponds to not altering the index.
    """
    def __init__(self):
        super(Processor, self).__init__(self.process)

    @abc.abstractmethod
    def process(self, data):
        pass


class ColumnsExtractor(Processor):
    """Extracts the specified columns from the input data."""
    def __init__(self, columns):
        super(ColumnsExtractor, self).__init__()
        self.columns = columns

    def process(self, data):
        if isinstance(data, np.ndarray):
            return data[:, self.columns]
        if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            return data[self.columns]
        if isinstance(data, list):
            if isinstance(self.columns, list):
                return [[d[col] for col in self.columns] for d in data]
            if isinstance(self.columns, tuple):
                return [tuple([d[col] for col in self.columns]) for d in data]
            return [d[self.columns] for d in data]
        raise ValueError(__UNSUPPORTED_DATA_TYPE_ERROR__ % type(data))


class DataTypeEncoder(Processor):
    def __init__(self, dtype):
        super(DataTypeEncoder, self).__init__()
        self.dtype = dtype

    def process(self, data):
        if isinstance(data, np.ndarray):
            return data.astype(self.dtype)
        raise ValueError(__UNSUPPORTED_DATA_TYPE_ERROR__ % type(data))


class ReshapeEncoder(Processor):
    def __init__(self, shape):
        super(ReshapeEncoder, self).__init__()
        self.shape = shape

    def process(self, data):
        if isinstance(data, np.ndarray):
            return data.reshape(self.shape)
        raise ValueError(__UNSUPPORTED_DATA_TYPE_ERROR__ % type(data))


class OneHotEncoder(Processor):
    def __init__(self, encoding_size, encoding_map=None):
        super(OneHotEncoder, self).__init__()
        self.encoding_size = encoding_size
        self.encoding_map = encoding_map

    def process(self, data):
        if isinstance(data, np.ndarray):
            num_samples = data.shape[0]
            encoded_array = np.zeros([num_samples, self.encoding_size])
            index_offset = np.arange(num_samples) * self.encoding_size
            if self.encoding_map is None:
                encoded_array.flat[index_offset + data.ravel()] = 1
                return encoded_array
            map_values = np.vectorize(lambda k: self.encoding_map[k])
            encoded_array.flat[index_offset + map_values(data.ravel())] = 1
            return encoded_array
        raise ValueError(__UNSUPPORTED_DATA_TYPE_ERROR__ % type(data))


class OneHotDecoder(Processor):
    def __init__(self, encoding_size, encoding_map=None, decoding_map=None):
        super(OneHotDecoder, self).__init__()
        self.encoding_size = encoding_size
        if encoding_map is None:
            self.decoding_map = decoding_map
        else:
            self.decoding_map = {v: k for k, v in encoding_map.items()}

    def process(self, data):
        if isinstance(data, np.ndarray):
            decoded_array = data.nonzero()[1]
            if self.decoding_map is None:
                return decoded_array
            map_values = np.vectorize(lambda v: self.decoding_map[v])
            return map_values(decoded_array)
        raise ValueError(__UNSUPPORTED_DATA_TYPE_ERROR__ % type(data))
