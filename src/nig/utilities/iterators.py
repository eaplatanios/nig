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

import itertools
import numpy as np

from math import factorial
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


class NonOverlappingCombinations(Iterator):
    """
    Returns p-length combinations of elements in {0,...,n-1} in the iterable,
    where the groups are non-overlapping. The groups are randomly generated.

    Example:
        NonOverlappingCombinations(7, 3) --> (0,5,2), (1,7,3)
    """
    def __init__(self, n, p, seed=None):
        assert p <= n
        # The total number of elements.
        self._n = n
        # The size of the groups.
        self._p = p
        # Initialize the random number generator.
        self._seed = seed
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()
        # Generate a random permutation of all n elements, out of which we will
        # take out p at a time.
        self._permutation = self._rng.permutation(n)
        self._start_index = 0
        # Keep track of the number of remaining combinations.
        self._remaining_combinations = n // p

    def next(self):
        if self._remaining_combinations <= 0:
            raise StopIteration()
        self._remaining_combinations -= 1
        self._start_index += self._p
        return self._permutation[self._start_index - self._p: self._start_index]

    def reset(self, n=None, p=None, seed=None):
        self._n = n if n is not None else self._n
        self._p = p if p is not None else self._p
        assert self._p <= self._n
        if seed and seed != self._seed:
            self._seed = seed
            self._rng = np.random.RandomState(self._seed)
            self._permutation = self._rng.permutation(n)
        self._start_index = 0
        self._remaining_combinations = self._n // self._p

    def reset_copy(self, n=None, p=None, seed=None):
        n = n if n is not None else self._n
        p = p if p is not None else self._p
        seed = seed if seed is not None else self._seed
        return NonOverlappingCombinations(n=n, p=p, seed=seed)

    def __len__(self):
        return self._n // self._p

    def remaining_length(self):
        return self._remaining_combinations


class AlmostRandomCombinations(Iterator):
    """
    Returns p-length combinations of elements in {0,...,n-1} in the iterable,
    where the groups are can be overlapping. The groups are randomly generated.
    This is similar to itertools.combinations, but itertools.combinations can
    only return the combinations in lexicographical order, starting with a
    provided configuration. However, calling the next operator on a Combinations
    object returns combinations in random order, under an approximately
    uniform distribution over n choose p.

    Example:

    """
    def __init__(self, n, p, max_combinations=None, seed=None):
        assert p <= n
        # The total number of elements.
        self._n = n
        # The size of the groups.
        self._p = p
        # Initialize the random number generator.
        self._seed = seed
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()
        self._combinations = itertools.combinations(range(n), p)

        # Keep track of the number of remaining combinations.
        self._total_combinations =  \
            int(factorial(n) / (factorial(n - p) * factorial(p)))

        # Keep track of the remaining combinations of the self._combinations
        # object.
        self._remaining_combinations = self._total_combinations

        # Set the max number of combinations to generate, after which we return
        # StopIteration()
        self._max_combinations = self._total_combinations \
            if max_combinations is None \
            else min(max_combinations, self._total_combinations)

        # Keep track of the number of combinations that still need to be generated
        # up to max_combinations.
        self._remaining_from_max = self._max_combinations

        # Set the rejection probability for rejection sampling.
        self._accept_probability = \
            self._max_combinations / self._total_combinations

    def next(self):
        if self._remaining_from_max <= 0:
            raise StopIteration()
        self._remaining_from_max -= 1

        combination = next(self._combinations)
        self._remaining_combinations -= 1
        while self._remaining_from_max < self._remaining_combinations and \
            self._rng.binomial(1, self._accept_probability) == 0:
            self._remaining_combinations -= 1
            combination = next(self._combinations)

        return combination

    def reset(self, n=None, p=None, max_combinations=None, seed=None):
        self._n = n if n is not None else self._n
        self._p = p if p is not None else self._p
        assert self._p <= self._n
        if seed and seed != self._seed:
            self._seed = seed
            self._rng = np.random.RandomState(self._seed)
        self._combinations = itertools.combinations(range(n), p)
        self._total_combinations = \
            int(factorial(n) / (factorial(n - p) * factorial(p)))
        self._remaining_combinations = self._total_combinations

        max_combinations = max_combinations \
            if max_combinations is not None else self._max_combinations
        self._max_combinations = self._total_combinations \
            if max_combinations is None \
            else min(max_combinations, self._total_combinations)

        self._remaining_from_max = self._max_combinations

    def reset_copy(self, n=None, p=None, max_combinations=None, seed=None):
        n = n if n is not None else self._n
        p = p if p is not None else self._p
        max_combinations = max_combinations \
            if max_combinations is not None else self._max_combinations
        seed = seed if seed is not None else self._seed
        return RandomCombinations(
            n=n, p=p, max_combinations=max_combinations, seed=seed)

    def __len__(self):
        return self._total_combinations

    def remaining_length(self):
        return self._remaining_from_max


class RandomCombinations(Iterator):
    """
    Returns p-length combinations of elements in {0,...,n-1} in the iterable,
    where the groups are can be overlapping. The groups are randomly generated.
    This is similar to itertools.combinations, but itertools.combinations can
    only return the combinations in lexicographical order, starting with a
    provided configuration. However, calling the next operator on a Combinations
    object returns combinations in random order, under a uniform distribution
    over n choose p.

    Example:

    """
    def __init__(self, n, p, max_combinations=None, seed=None):
        assert p <= n
        # The total number of elements.
        self._n = n
        # The size of the groups.
        self._p = p
        # Initialize the random number generator.
        self._seed = seed
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()

        # Keep track of the number of remaining combinations.
        self._total_combinations = \
            int(factorial(n) / (factorial(n - p) * factorial(p)))

        # Set the max number of combinations to generate, after which we return
        # StopIteration()
        self._max_combinations = self._total_combinations \
            if max_combinations is None \
            else min(max_combinations, self._total_combinations)

        # Keep track of the remaining combinations.
        self._counter = 0

        # Generate the combinations in random order, and store only the first
        # self._max_combinations.
        self._combinations = list(itertools.combinations(range(n), p))
        self._rng.shuffle(self._combinations)
        self._combinations = self._combinations[:max_combinations]

    def next(self):
        if self._counter >= self._max_combinations:
            raise StopIteration()
        self._counter += 1
        return self._combinations[self._counter-1]

    def reset(self, n=None, p=None, max_combinations=None, seed=None):
        self._n = n if n is not None else self._n
        self._p = p if p is not None else self._p
        assert self._p <= self._n
        if seed and seed != self._seed:
            self._seed = seed
            self._rng = np.random.RandomState(self._seed)
        self._total_combinations = \
            int(factorial(n) / (factorial(n - p) * factorial(p)))
        self._remaining_combinations = self._total_combinations

        max_combinations = max_combinations \
            if max_combinations is not None else self._max_combinations
        self._max_combinations = self._total_combinations \
            if max_combinations is None \
            else min(max_combinations, self._total_combinations)

        self._counter = 0

        # Generate the combinations in random order, and store only the first
        # self._max_combinations.
        self._combinations = list(itertools.combinations(range(n), p))
        self._rng.shuffle(self._combinations)
        self._combinations = self._combinations[:max_combinations]

    def reset_copy(self, n=None, p=None, max_combinations=None, seed=None):
        n = n if n is not None else self._n
        p = p if p is not None else self._p
        max_combinations = max_combinations \
            if max_combinations is not None else self._max_combinations
        seed = seed if seed is not None else self._seed
        return RandomCombinations(
            n=n, p=p, max_combinations=max_combinations, seed=seed)

    def __len__(self):
        return self._total_combinations

    def remaining_length(self):
        return self._max_combinations - self._counter
