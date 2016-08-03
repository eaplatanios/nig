from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np

from nig.utilities import pipe

__author__ = 'Emmanouil Antonios Platanios'


class Iterator(object):
    __metaclass__ = abc.ABCMeta

    def __iter__(self):
        return self

    # This function is added for compatibility with Python 3
    def __next__(self):
        return self.next()

    @abc.abstractmethod
    def next(self):
        pass

    @abc.abstractmethod
    def reset(self, batch_size=None, shuffle=None, cycle=None,
              cycle_shuffle=None, keep_last_batch=None):
        pass

    @abc.abstractmethod
    def reset_copy(self, batch_size=None, shuffle=None, cycle=None,
                   cycle_shuffle=None, keep_last_batch=None):
        pass


class BaseDataIterator(Iterator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, batch_size, shuffle=False, cycle=False,
                 cycle_shuffle=False, keep_last_batch=False, pipelines=None):
        self.data = data
        self.batch_size = batch_size
        if shuffle:
            self.shuffle_data()
        self.shuffle = shuffle
        self.cycle = cycle
        self.cycle_shuffle = cycle_shuffle
        self.keep_last_batch = keep_last_batch
        self.pipelines = self._preprocess_pipelines(None, pipelines)
        self._total_length = len(data)
        self._begin_index = 0
        self._end_index = -1
        self._reached_end = False

    @staticmethod
    def _preprocess_pipelines(current_pipelines, new_pipelines):
        if new_pipelines is not None:
            if type(new_pipelines) is list:
                return [(lambda x: x) if pipeline is None
                        else pipeline if callable(pipeline)
                        else pipe(*pipeline)
                        for pipeline in new_pipelines]
            else:
                return [pipeline if callable(pipeline)
                        else pipe(*pipeline)
                        for pipeline in new_pipelines]
        elif current_pipelines is not None:
            return current_pipelines
        else:
            return [lambda x: x]

    def next(self):
        next_data = None
        if self.cycle or not self._reached_end:
            self._end_index = self._begin_index + self.batch_size
            if self._end_index > self._total_length - 1:
                self._reached_end = True
            begin_index = self._begin_index
            self._begin_index += self.batch_size
            if not self._reached_end:
                next_data = self.get_data(begin_index, self._end_index)
            elif self.cycle:
                self._begin_index %= self._total_length
                if self.cycle_shuffle:
                    self.shuffle_data()
                self._reached_end = False
                end_index = self._end_index - self._total_length
                next_data = self.concatenate_data(
                    self.get_data(begin_index, -1),
                    self.get_data(0, end_index)
                )
            elif self.keep_last_batch:
                next_data = self.get_data(begin_index, -1)
        if next_data is not None:
            return tuple(pipeline(next_data) for pipeline in self.pipelines)
        raise StopIteration()

    @abc.abstractmethod
    def shuffle_data(self):
        pass

    @abc.abstractmethod
    def get_data(self, from_index, to_index):
        pass

    @abc.abstractmethod
    def concatenate_data(self, data_batch_1, data_batch_2):
        pass

    def reset(self, batch_size=None, shuffle=None, cycle=None,
              cycle_shuffle=None, keep_last_batch=None, pipelines=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if shuffle is not None:
            self.shuffle = shuffle
        if cycle is not None:
            self.cycle = cycle
        if cycle_shuffle is not None:
            self.cycle_shuffle = cycle_shuffle
        if keep_last_batch is not None:
            self.keep_last_batch = keep_last_batch
        self.pipelines = self._preprocess_pipelines(self.pipelines, pipelines)
        if self.shuffle:
            self.shuffle_data()
        self._begin_index = 0
        self._end_index = -1
        self._reached_end = False

    def reset_copy(self, batch_size=None, shuffle=None, cycle=None,
                   cycle_shuffle=None, keep_last_batch=None, pipelines=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        shuffle = shuffle if shuffle is not None else self.shuffle
        cycle = cycle if cycle is not None else self.cycle
        cycle_shuffle = cycle_shuffle if cycle_shuffle is not None \
            else self.cycle_shuffle
        keep_last_batch = keep_last_batch if keep_last_batch is not None \
            else self.keep_last_batch
        pipelines = pipelines if pipelines is not None else self.pipelines
        return self.__class__(self.data, batch_size, shuffle, cycle,
                              cycle_shuffle, keep_last_batch, pipelines)


class NPArrayIterator(BaseDataIterator):
    def shuffle_data(self):
        np.random.shuffle(self.data)

    def get_data(self, from_index, to_index):
        return self.data[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return np.vstack([data_batch_1, data_batch_2])
