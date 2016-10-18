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

from itertools import combinations
from math import factorial
from six import with_metaclass

from ...utilities.iterators import Iterator

__author__ = 'eaplatanios'


class CrossValidation(with_metaclass(abc.ABCMeta, Iterator)):
    def __init__(self, data_size, shuffle=False, seed=None):
        self.data_size = data_size
        self.shuffle = shuffle
        if self.shuffle:
            if isinstance(seed, np.random.RandomState):
                self.rng = seed
            else:
                self.rng = np.random.RandomState(seed)
            self.indices = self.rng.permutation(np.arange(data_size))
        else:
            self.indices = np.arange(data_size)

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


class _Base(with_metaclass(abc.ABCMeta, CrossValidation)):
    @abc.abstractmethod
    def _next_test(self):
        pass

    def next(self):
        test_mask = self._empty_mask()
        test_mask[self._next_test()] = True
        train_mask = np.logical_not(test_mask)
        return self.indices[train_mask], self.indices[test_mask]

    def _empty_mask(self):
        return np.zeros(self.data_size, dtype=np.bool)

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


class LeaveOneOut(_Base):
    def __init__(self, data_size):
        super(LeaveOneOut, self).__init__(data_size, shuffle=False)
        self._current_index = -1

    def _next_test(self):
        self._current_index += 1
        if self._current_index > self.data_size - 1:
            raise StopIteration()
        return self._current_index

    def reset(self):
        self._current_index = -1

    def reset_copy(self):
        return LeaveOneOut(self.data_size)

    def __len__(self):
        return self.data_size

    def remaining_length(self):
        return self.data_size - self._current_index - 1


class LeavePOut(_Base):
    def __init__(self, data_size, p=1):
        super(LeavePOut, self).__init__(data_size, shuffle=False)
        self.p = p
        self._test_combinations = combinations(range(data_size), p)
        self._total_combinations = int(
            factorial(data_size) / (factorial(data_size - p) * factorial(p)))
        self._remaining_combinations = self._total_combinations

    def _next_test(self):
        if self._remaining_combinations <= 0:
            raise StopIteration()
        self._remaining_combinations -= 1
        return np.array(self._test_combinations.next())

    def reset(self, p=None):
        if p is not None:
            self.p = p
        self._test_combinations = combinations(range(self.data_size), self.p)
        self._total_combinations = int(
            factorial(self.data_size)
            / (factorial(self.data_size - self.p) * factorial(self.p)))
        self._remaining_combinations = self._total_combinations

    def reset_copy(self, p=None):
        p = p if p is not None else self.p
        return LeavePOut(self.data_size, p)

    def __len__(self):
        return self._total_combinations

    def remaining_length(self):
        return self._remaining_combinations


class LeaveOneLabelOut(_Base):
    def __init__(self, labels):
        super(LeaveOneLabelOut, self).__init__(len(labels), shuffle=False)
        self.labels = np.asarray(labels)
        self.unique_labels = np.unique(labels)
        self.num_unique_labels = len(self.unique_labels)
        self._current_label_index = -1

    def _next_test(self):
        self._current_label_index += 1
        if self._current_label_index > self.num_unique_labels - 1:
            raise StopIteration()
        return self.labels == self.unique_labels[self._current_label_index]

    def reset(self):
        self._current_label_index = -1

    def reset_copy(self):
        return LeaveOneLabelOut(self.labels)

    def __len__(self):
        return self.num_unique_labels

    def remaining_length(self):
        return self.num_unique_labels - self._current_label_index - 1


class LeavePLabelsOut(_Base):
    def __init__(self, labels, p=1):
        super(LeavePLabelsOut, self).__init__(len(labels), shuffle=False)
        self.labels = np.asarray(labels)
        self.unique_labels = np.unique(labels)
        self.num_unique_labels = len(self.unique_labels)
        self.p = p
        self._test_combinations = combinations(range(self.num_unique_labels), p)
        self._total_combinations = int(
            factorial(self.num_unique_labels)
            / (factorial(self.num_unique_labels - p) * factorial(p)))
        self._remaining_combinations = self._total_combinations

    def _next_test(self):
        pass

    def next(self):
        if self._remaining_combinations <= 0:
            raise StopIteration()
        self._remaining_combinations -= 1
        test_mask = self._empty_mask()
        test_combination = np.array(self._test_combinations.next())
        for label in self.unique_labels[test_combination]:
            test_mask[self.labels == label] = True
        train_mask = np.logical_not(test_mask)
        return self.indices[train_mask], self.indices[test_mask]

    def reset(self, p=None):
        if p is not None:
            self.p = p
        self._test_combinations = combinations(
            range(self.num_unique_labels), self.p)
        self._total_combinations = int(
            factorial(self.num_unique_labels)
            / (factorial(self.num_unique_labels - self.p) * factorial(self.p)))
        self._remaining_combinations = self._total_combinations

    def reset_copy(self, p=None):
        p = p if p is not None else self.p
        return LeavePLabelsOut(self.labels, p)

    def __len__(self):
        return self._total_combinations

    def remaining_length(self):
        return self._remaining_combinations


class _KFoldBase(with_metaclass(abc.ABCMeta, _Base)):
    def __init__(self, data_size, k, shuffle=False, seed=None):
        if k <= 1:
            raise ValueError('The number of folds, k, needs to be > 1. '
                             'Provided value is %d.' % k)
        if k > data_size:
            raise ValueError('The number of folds, k, cannot be larger than '
                             'the number of data instances. Provided k is %d '
                             'and the number of data instances is %d.'
                             % (k, data_size))
        super(_KFoldBase, self).__init__(data_size, shuffle, seed)
        self.k = k

    @abc.abstractmethod
    def _next_test(self):
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


class KFold(_KFoldBase):
    def __init__(self, data_size, k, shuffle=False, seed=None):
        super(KFold, self).__init__(data_size, k, shuffle, seed)
        self.fold_sizes = (data_size // k) * np.ones(k, dtype=np.int)
        self.fold_sizes[:data_size % k] += 1
        self._current_fold = 0
        self._current_index = 0

    def _next_test(self):
        if self._current_fold >= self.k:
            raise StopIteration()
        previous_index = self._current_index
        previous_fold = self._current_fold
        self._current_fold += 1
        self._current_index += self.fold_sizes[previous_fold]
        return self.indices[previous_index:self._current_index]

    def reset(self, k=None, shuffle=None, seed=None):
        if k is not None:
            self.k = k
        if shuffle is not None:
            self.shuffle = shuffle
        if self.shuffle:
            if seed is not None:
                self.rng = np.random.RandomState(seed)
            self.indices = self.rng.permutation(np.arange(self.data_size))
        else:
            self.indices = np.arange(self.data_size)
        self.fold_sizes = \
            (self.data_size // self.k) * np.ones(self.k, dtype=np.int)
        self.fold_sizes[:self.data_size % self.k] += 1
        self._current_fold = 0
        self._current_index = 0

    def reset_copy(self, k=None, shuffle=None, seed=None):
        k = k if k is not None else self.k
        shuffle = shuffle if shuffle is not None else self.shuffle
        return KFold(self.data_size, k, shuffle, seed)

    def __len__(self):
        return self.k

    def remaining_length(self):
        return self.k - self._current_fold


class StratifiedKFold(_KFoldBase):
    def __init__(self, labels, k, shuffle=False, seed=None):
        super(StratifiedKFold, self).__init__(len(labels), k, shuffle, seed)
        self.labels = np.asarray(labels)
        self.test_folds = None
        self._current_fold = 0
        self._initialize()

    def _initialize(self):
        labels = self.labels[self.indices]
        unique_labels, indices = np.unique(labels, return_inverse=True)
        label_counts = np.bincount(indices)
        min_label_count = np.min(label_counts)
        if self.k > min_label_count:
            raise ValueError('The least frequent label appears only %d times, '
                             'which is lower than the number of folds %d. '
                             'Stratified k-fold cross-validation is thus not '
                             'possible.' % (min_label_count, self.k))
        # We use individual KFold cross-validation strategies for each label in
        # order to respect the balance of the labels.
        per_label_k_folds = [KFold(
            count, k=self.k, shuffle=self.shuffle, seed=self.rng)
                             for count in label_counts]
        test_folds = np.zeros(self.data_size, dtype=np.int)
        for fold_index, per_label_k_fold in enumerate(zip(*per_label_k_folds)):
            for label, (_, test_split) in zip(unique_labels, per_label_k_fold):
                label_test_folds = test_folds[labels == label]
                label_test_folds[test_split] = fold_index
                test_folds[labels == label] = label_test_folds
        self.test_folds = test_folds
        self._current_fold = 0

    def _next_test(self):
        pass

    def next(self):
        if self._current_fold == self.k:
            raise StopIteration()
        test_mask = self._empty_mask()
        test_mask[self.test_folds == self._current_fold] = True
        train_mask = np.logical_not(test_mask)
        self._current_fold += 1
        return self.indices[train_mask], self.indices[test_mask]

    def reset(self, k=None, shuffle=None, seed=None):
        if k is not None:
            self.k = k
        if shuffle is not None:
            self.shuffle = shuffle
        if self.shuffle:
            if seed is not None:
                self.rng = np.random.RandomState(seed)
            self.indices = self.rng.permutation(np.arange(self.data_size))
        else:
            self.indices = np.arange(self.data_size)
        self._current_fold = 0
        self._initialize()

    def reset_copy(self, k=None, shuffle=None, seed=None):
        k = k if k is not None else self.k
        shuffle = shuffle if shuffle is not None else self.shuffle
        return StratifiedKFold(self.labels, k, shuffle, seed)

    def __len__(self):
        return self.k

    def remaining_length(self):
        return self.k - self._current_fold


# TODO: Add LabelKFold.
# TODO: Add ShuffleSplit.
# TODO: Add LabelShuffleSplit.
# TODO: Add PredefinedSplit.
