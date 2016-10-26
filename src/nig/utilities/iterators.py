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

from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = ['Iterator', 'ZipIterator']


class Iterator(with_metaclass(abc.ABCMeta, object)):
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


class ZipIterator(Iterator):
    def __init__(self, iterators, keys=None):
        if any(len(iterator) != len(iterators[0]) for iterator in iterators):
            raise ValueError('The iterators being zipped must all have equal '
                             'length.')
        self._iterators = iterators
        if keys is not None:
            if len(iterators) != len(keys):
                raise ValueError('The number of iterators %d does not match '
                                 'the number of keys %d.'
                                 % (len(iterators), len(keys)))
        self._keys = keys

    def next(self):
        if self._keys is None:
            return tuple([iterator.next() for iterator in self._iterators])
        return {self._keys[i]: iterator.next()
                for i, iterator in enumerate(self._iterators)}

    def reset(self):
        for iterator in self._iterators:
            iterator.reset()
        return self

    def reset_copy(self):
        return ZipIterator([iterator.reset_copy()
                            for iterator in self._iterators], self._keys)

    def __len__(self):
        return len(self._iterators[0])

    def remaining_length(self):
        return self._iterators[0].remaining_length()
