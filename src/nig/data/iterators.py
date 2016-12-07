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

from nig.utilities.functions import pipe
from nig.utilities.iterators import Iterator

__author__ = 'eaplatanios'

__all__ = ['get_iterator', 'DataIterator', 'ListIterator', 'NPArrayIterator',
           'PDDataFrameIterator', 'ZipDataIterator']


def get_iterator(data, batch_size=None, shuffle=False, cycle=False,
                 cycle_shuffle=False, keep_last=True, pipelines=None):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else data.shape[0]
        return NPArrayIterator(
            data, batch_size=batch_size, shuffle=shuffle, cycle=cycle,
            cycle_shuffle=cycle_shuffle, keep_last=keep_last,
            pipelines=pipelines)
    if isinstance(data, tuple):
        # TODO: Add shuffling capability.
        return ZipDataIterator(
            iterators=[_process_data_element(
                data=d, batch_size=batch_size, keep_last=keep_last)
                       for d in data],
            keys=None, batch_size=batch_size, cycle=cycle, pipelines=pipelines)
    if isinstance(data, dict):
        # TODO: Add shuffling capability.
        if isinstance(pipelines, dict):
            pipelines = [pipelines[k] for k in data.keys()]
        return ZipDataIterator(
            iterators=[_process_data_element(
                data=d, batch_size=batch_size, keep_last=keep_last)
                       for d in data.values()],
            keys=list(data.keys()), batch_size=batch_size, cycle=cycle,
            pipelines=pipelines)
    if not isinstance(data, DataIterator) \
            and not isinstance(data, ZipDataIterator):
        raise TypeError('Unsupported data type %s encountered.' % type(data))
    return data.reset_copy(
        batch_size=batch_size, shuffle=shuffle, cycle=cycle,
        cycle_shuffle=cycle_shuffle, keep_last=keep_last, pipelines=pipelines)


def _process_data_element(data, batch_size=None, cycle=False, keep_last=True):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else data.shape[0]
        return NPArrayIterator(
            data=data, batch_size=batch_size, cycle=cycle, keep_last=keep_last)
    if not isinstance(data, DataIterator):
        raise TypeError('Unsupported data type %s encountered.' % type(data))
    return data.reset_copy(
        batch_size=batch_size, cycle=cycle, keep_last=keep_last)


def _process_pipelines(pipelines):
    if pipelines is None:
        return lambda x: x
    if callable(pipelines):
        return pipelines
    if isinstance(pipelines, list):
        return [(lambda x: x) if pipeline is None
                else pipeline if callable(pipeline)
                else pipe(*pipeline)
                for pipeline in pipelines]
    if isinstance(pipelines, tuple):
        return tuple([(lambda x: x) if pipeline is None
                      else pipeline if callable(pipeline)
                      else pipe(*pipeline)
                      for pipeline in pipelines])
    if isinstance(pipelines, dict):
        return {k: (lambda x: x) if pipeline is None
                else pipeline if callable(pipeline)
                else pipe(*pipeline)
                for k, pipeline in pipelines.items()}
    raise TypeError('Unsupported pipelines type.')


def _apply_pipelines(data, pipelines):
    if callable(pipelines):
        return pipelines(data)
    if isinstance(pipelines, list):
        return [pipeline(data) for pipeline in pipelines]
    if isinstance(pipelines, tuple):
        return tuple([pipeline(data) for pipeline in pipelines])
    if isinstance(pipelines, dict):
        return {k: pipeline(data) for k, pipeline in pipelines.items()}
    raise TypeError('Unsupported pipelines type.')


# TODO: Add support for various last batch filling options (e.g., pad, roll).
# TODO: Add support for a CSV iterator.
# TODO: Add support for a prefetching iterator.

class DataIterator(with_metaclass(abc.ABCMeta, Iterator)):
    """Constructs and returns a data iterator to be used with a learner for
    the specified data.

    Args:
        data:
        batch_size (int): Optional batch size value. Defaults to 128.
        shuffle (bool): Optional boolean value indicating whether to shuffle
            the data instances before iterating over them. Defaults to
            `False`.
        cycle (bool): Optional boolean value indicating whether the returned
            iterator should cycle when it reaches its end and become an
            effectively infinite iterator. Defaults to `False`.
        cycle_shuffle (bool): Optional boolean value indicating whether the
            returned iterator should shuffle the data instances at the end
            of each cycle. Defaults to `False`. Note that this argument is
            only effective if `cycle` is set to `True`.
        keep_last (bool): Optional boolean value indicating whether the
            returned iterator should keep and return the last batch in the
            data, if that batch has size less than the specified batch size.
        seed (long): Optional seed value for the random number generator
            used when shuffling the data within the iterator.

    Returns:
        Iterator: Constructed iterator to be used with learners.
    """
    def __init__(self, data, batch_size=128, shuffle=False, cycle=False,
                 cycle_shuffle=False, keep_last=True, pipelines=None,
                 seed=None):
        self._length = len(data)
        self.data = data
        batch_size = batch_size if batch_size is not None else self._length
        self.batch_size = min(batch_size, self._length)
        self.shuffle = shuffle
        self.cycle = cycle
        self.cycle_shuffle = cycle_shuffle
        self.keep_last = keep_last
        self.pipelines = _process_pipelines(pipelines)
        self.rng = np.random.RandomState(seed)
        self._begin_index = 0
        self._reached_end = False
        if self.shuffle:
            self.shuffle_data()

    def next(self):
        next_data = None
        if self.cycle or not self._reached_end:
            begin_index = self._begin_index
            self._begin_index += self.batch_size
            if self._begin_index > self._length:
                self._reached_end = True
            if not self._reached_end:
                next_data = self.get_data(begin_index, self._begin_index)
            elif self.cycle:
                if begin_index < self._length:
                    data_batch_1 = self.get_data(begin_index, None)
                    if self.cycle_shuffle:
                        self.shuffle_data()
                    next_data = self.concatenate_data(
                        data_batch_1,
                        self.get_data(0, self._begin_index - self._length))
                else:
                    if self.cycle_shuffle:
                        self.shuffle_data()
                    next_data = self.get_data(0, self.batch_size)
                self._reached_end = False
                self._begin_index %= self._length
            elif self.keep_last and begin_index != self._length:
                next_data = self.get_data(begin_index, None)
        if next_data is not None:
            return _apply_pipelines(next_data, self.pipelines)
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
            self.batch_size = min(batch_size, self._length)
        if shuffle is not None:
            self.shuffle = shuffle
        if cycle is not None:
            self.cycle = cycle
        if cycle_shuffle is not None:
            self.cycle_shuffle = cycle_shuffle
        if keep_last is not None:
            self.keep_last = keep_last
        if pipelines is not None:
            self.pipelines = _process_pipelines(pipelines)
        if self.shuffle:
            self.shuffle_data()
        self._begin_index = 0
        self._reached_end = False
        return self

    def reset_copy(self, batch_size=None, shuffle=None, cycle=None,
                   cycle_shuffle=None, keep_last=None, pipelines=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        batch_size = min(batch_size, self._length)
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
        return self._length


class ListIterator(DataIterator):
    def shuffle_data(self):
        indices = self.rng.permutation(np.arange(self._length))
        self.data = self._index(self.data, indices)

    @staticmethod
    def _index(array, indices):
        return [array[i] for i in indices]

    def get_data(self, from_index, to_index):
        return self.data[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return data_batch_1 + data_batch_2


class NPArrayIterator(DataIterator):
    def shuffle_data(self):
        indices = self.rng.permutation(np.arange(self._length))
        self.data = self.data[indices]

    def get_data(self, from_index, to_index):
        return self.data[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return np.concatenate([data_batch_1, data_batch_2], axis=0)


class PDDataFrameIterator(DataIterator):
    def shuffle_data(self):
        indices = self.rng.permutation(np.arange(self._length))
        self.data = self.data.iloc[indices]

    def get_data(self, from_index, to_index):
        return self.data.iloc[from_index:to_index]

    def concatenate_data(self, data_batch_1, data_batch_2):
        return pd.concat([data_batch_1, data_batch_2])


class ZipDataIterator(Iterator):
    # TODO: Handle shuffling operations correctly.
    def __init__(self, iterators, keys=None, batch_size=128, shuffle=False,
                 cycle=False, cycle_shuffle=False, keep_last=True,
                 pipelines=None, seed=None):
        self._length = len(iterators[0])
        if any(len(it) != self._length for it in iterators):
            raise ValueError('The iterators being zipped must all have equal '
                             'length.')
        if any(not isinstance(it, DataIterator) for it in iterators):
            raise TypeError('The iterators being zipped must be DataIterators.')
        if keys is not None:
            if len(iterators) != len(keys):
                raise ValueError('The number of iterators %d does not match '
                                 'the number of keys %d.'
                                 % (len(iterators), len(keys)))
        if pipelines is not None:
            if len(iterators) != len(pipelines):
                raise ValueError('The number of iterators %d does not match '
                                 'the number of pipelines %d.'
                                 % (len(iterators), len(pipelines)))
        else:
            pipelines = [it.pipelines for it in iterators]
        if shuffle or cycle_shuffle:
            raise NotImplementedError('Shuffling is not currently supported '
                                      'for zip iterators.')
        batch_size = batch_size if batch_size is not None else self._length
        self.batch_size = min(batch_size, self._length)
        self.shuffle = shuffle
        self.cycle = cycle
        self.cycle_shuffle = cycle_shuffle
        self.keep_last = keep_last
        self.pipelines = pipelines
        self.seed = seed
        self._iterators = iterators
        self._keys = keys
        self._reset_iterators()

    def _reset_iterators(self):
        self._iterators = [it.reset(
            batch_size=self.batch_size, shuffle=self.shuffle,
            cycle=self.cycle, cycle_shuffle=self.cycle_shuffle,
            keep_last=self.keep_last, pipelines=p)
                           for it, p in zip(self._iterators, self.pipelines)]

    def next(self):
        if self._keys is None:
            return tuple([iterator.next() for iterator in self._iterators])
        return {self._keys[i]: iterator.next()
                for i, iterator in enumerate(self._iterators)}

    def reset(self, batch_size=None, shuffle=None, cycle=None,
              cycle_shuffle=None, keep_last=None, pipelines=None):
        if batch_size is not None:
            self.batch_size = min(batch_size, self._length)
        if shuffle is not None:
            self.shuffle = shuffle
        if cycle is not None:
            self.cycle = cycle
        if cycle_shuffle is not None:
            self.cycle_shuffle = cycle_shuffle
        if keep_last is not None:
            self.keep_last = keep_last
        if pipelines is not None:
            self.pipelines = pipelines
        if self.shuffle or self.cycle_shuffle:
            raise NotImplementedError('Shuffling is not currently supported '
                                      'for zip iterators.')
        self._reset_iterators()
        return self

    def reset_copy(self, batch_size=None, shuffle=None, cycle=None,
                   cycle_shuffle=None, keep_last=None, pipelines=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        batch_size = min(batch_size, self._length)
        shuffle = shuffle if shuffle is not None else self.shuffle
        cycle = cycle if cycle is not None else self.cycle
        cycle_shuffle = cycle_shuffle if cycle_shuffle is not None \
            else self.cycle_shuffle
        keep_last = keep_last if keep_last is not None else self.keep_last
        pipelines = pipelines if pipelines is not None else self.pipelines
        return ZipDataIterator(
            iterators=self._iterators, keys=self._keys, batch_size=batch_size,
            shuffle=shuffle, cycle=cycle, cycle_shuffle=cycle_shuffle,
            keep_last=keep_last, pipelines=pipelines)

    def remaining_length(self):
        # TODO: Confirm that all iterators have the same remaining length.
        return self._iterators[0].remaining_length()

    def __len__(self):
        return self._length
