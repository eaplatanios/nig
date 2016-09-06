from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np
import pandas as pd

from nig.functions import pipe

__author__ = 'eaplatanios'


class Iterator(object):
    __metaclass__ = abc.ABCMeta

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abc.abstractmethod
    def next(self):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset_copy(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def remaining_length(self):
        pass


class BaseDataIterator(Iterator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, batch_size=128, shuffle=False, cycle=False,
                 cycle_shuffle=False, keep_last=False, pipelines=None,
                 seed=None):
        self.data = data
        self.batch_size = batch_size
        if shuffle:
            self.shuffle_data()
        self.shuffle = shuffle
        self.cycle = cycle
        self.cycle_shuffle = cycle_shuffle
        self.keep_last = keep_last
        self.pipelines = self._preprocess_pipelines(None, pipelines)
        self.rng = np.random.RandomState(seed)
        if isinstance(self.data, tuple):
            self._total_length = len(self.data[0])
            for d in self.data:
                if len(d) != self._total_length:
                    raise ValueError('All tuple elements must have the same '
                                     'length.')
        else:
            self._total_length = len(self.data)
        self._begin_index = 0
        self._reached_end = False

    @staticmethod
    def _preprocess_pipelines(current_pipelines, new_pipelines):
        if new_pipelines is not None:
            if isinstance(new_pipelines, list):
                return [(lambda x: x) if pipeline is None
                        else pipeline if callable(pipeline)
                        else pipe(*pipeline)
                        for pipeline in new_pipelines]
            return [new_pipelines]
        elif current_pipelines is not None:
            return current_pipelines
        return [lambda x: x]

    def next(self):
        next_data = None
        if self.cycle or not self._reached_end:
            begin_index = self._begin_index
            self._begin_index += self.batch_size
            if self._begin_index > self._total_length:
                self._reached_end = True
            if not self._reached_end:
                next_data = self.get_data(begin_index, self._begin_index)
            elif self.cycle:
                if self.cycle_shuffle:
                    self.shuffle_data()
                self._reached_end = False
                next_data = self.concatenate_data(
                    self.get_data(begin_index, -1),
                    self.get_data(0, self._begin_index - self._total_length)
                )
                self._begin_index %= self._total_length
            elif self.keep_last and begin_index != self._total_length:
                next_data = self.get_data(begin_index, None)
        if next_data is not None:
            next_data = [pipeline(next_data) for pipeline in self.pipelines]
            return tuple(next_data) if len(next_data) > 1 else next_data[0]
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
              cycle_shuffle=None, keep_last=None, pipelines=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if shuffle is not None:
            self.shuffle = shuffle
        if cycle is not None:
            self.cycle = cycle
        if cycle_shuffle is not None:
            self.cycle_shuffle = cycle_shuffle
        if keep_last is not None:
            self.keep_last = keep_last
        self.pipelines = self._preprocess_pipelines(self.pipelines, pipelines)
        if self.shuffle:
            self.shuffle_data()
        self._begin_index = 0
        self._reached_end = False

    def reset_copy(self, batch_size=None, shuffle=None, cycle=None,
                   cycle_shuffle=None, keep_last=None, pipelines=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        shuffle = shuffle if shuffle is not None else self.shuffle
        cycle = cycle if cycle is not None else self.cycle
        cycle_shuffle = cycle_shuffle if cycle_shuffle is not None \
            else self.cycle_shuffle
        keep_last = keep_last if keep_last is not None else self.keep_last
        pipelines = pipelines if pipelines is not None else self.pipelines
        return self.__class__(self.data, batch_size, shuffle, cycle,
                              cycle_shuffle, keep_last, pipelines)

    def remaining_length(self):
        return len(self) - self._begin_index if not self.cycle else -1

    def __len__(self):
        return self._total_length


class NPArrayIterator(BaseDataIterator):
    def shuffle_data(self):
        indices = self.rng.permutation(np.arange(self._total_length))
        self.data = tuple(data[indices] for data in self.data) \
            if isinstance(self.data, tuple) \
            else self.data[indices]

    def get_data(self, from_index, to_index):
        return tuple(data[from_index:to_index] for data in self.data) \
            if isinstance(self.data, tuple) \
            else self.data[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return tuple(np.vstack([db_1, db_2])
                     for (db_1, db_2) in zip(data_batch_1, data_batch_2)) \
            if isinstance(self.data, tuple) \
            else np.vstack([data_batch_1, data_batch_2])


class PDDataFrameIterator(BaseDataIterator):
    def shuffle_data(self):
        indices = self.rng.permutation(np.arange(self._total_length))
        self.data = tuple(data.iloc[indices] for data in self.data) \
            if isinstance(self.data, tuple) \
            else self.data.iloc[indices]

    def get_data(self, from_index, to_index):
        return tuple(data.iloc[from_index:to_index] for data in self.data) \
            if isinstance(self.data, tuple) \
            else self.data.iloc[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return tuple(pd.concat([db_1, db_2])
                     for (db_1, db_2) in zip(data_batch_1, data_batch_2)) \
            if isinstance(self.data, tuple) \
            else pd.concat([data_batch_1, data_batch_2])
