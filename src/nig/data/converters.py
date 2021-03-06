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

from six import with_metaclass

from nig.utilities.functions import PipelineFunction

__author__ = 'eaplatanios'

__all__ = ['Converter', 'TupleToDictConverter', 'ListToNPArrayConverter',
           'PDSeriesToNPConverter', 'PDDataFrameToNPConverter']


class Converter(with_metaclass(abc.ABCMeta, PipelineFunction)):
    """A :class:`Converter` receives some data as input and provides a function
    for transforming that data to a new format.
    """
    def __init__(self):
        super(Converter, self).__init__(self.convert)

    @abc.abstractmethod
    def convert(self, data):
        pass


class TupleToDictConverter(Converter):
    def __init__(self, keys):
        super(TupleToDictConverter, self).__init__()
        self.keys = keys

    def convert(self, data):
        return dict(zip(self.keys, data))


class ListToNPArrayConverter(Converter):
    def __init__(self, flatten=False):
        super(ListToNPArrayConverter, self).__init__()
        self.flatten = flatten

    def convert(self, data):
        if not self.flatten:
            return np.array(data)
        return np.array([array.flatten(order='C') for array in data])


class PDSeriesToNPConverter(Converter):
    def __init__(self, data=None, periods=None, frequency=None, flatten=False):
        super(PDSeriesToNPConverter, self).__init__()
        if periods is None:
            periods = []
        self.data = data
        self.periods = periods if isinstance(periods, list) else [periods]
        self.frequency = frequency
        if flatten:
            self.__series_values_to_np = lambda values: np.asarray(
                    [np.asarray(value).flatten() for value in values])
        else:
            self.__series_values_to_np = lambda values: np.asarray(values)

    def convert(self, data):
        a_data = self.data if data is not None else data
        array = self.__series_values_to_np(data.values)
        if not self.periods:
            if len(array.shape) == 1:
                return array[:, np.newaxis]
            return array
        period_arrays = \
            [self.__series_values_to_np(
                a_data.shift(period, self.frequency).reindex(data.index).values)
             for period in self.periods]
        return np.concatenate([array] + period_arrays, axis=1)


class PDDataFrameToNPConverter(Converter):
    def __init__(self, data=None, columns=None, periods=None, frequency=None,
                 flatten=False):
        super(PDDataFrameToNPConverter, self).__init__()
        if isinstance(columns, list):
            columns = columns if len(columns) > 1 else columns[0]
        if periods is None:
            periods = []
        if data is not None and columns is not None:
            self.data = data[columns]
        else:
            self.data = data
        self.columns = columns
        self.periods = periods if isinstance(periods, list) else [periods]
        self.frequency = frequency
        if flatten and isinstance(columns, list):
            self.__data_frame_values_to_np = lambda values: np.asarray(
                [np.concatenate([np.asarray(v).flatten() for v in value])
                 for value in values])
        elif flatten:
            self.__data_frame_values_to_np = lambda values: np.asarray(
                [np.asarray(value).flatten() for value in values])
        else:
            self.__data_frame_values_to_np = lambda values: np.asarray(values)

    def convert(self, data):
        data = data if self.columns is None else data[self.columns]
        a_data = self.data if self.data is not None else data
        array = self.__data_frame_values_to_np(data.values)
        if not self.periods:
            if len(array.shape) == 1:
                return array[:, np.newaxis]
            return array
        period_arrays = \
            [self.__data_frame_values_to_np(
                a_data.shift(period, self.frequency).reindex(data.index).values)
             for period in self.periods]
        return np.concatenate([array] + period_arrays, axis=1)
