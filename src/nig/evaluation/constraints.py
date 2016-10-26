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

__all__ = ['Constraint', 'MutualExclusionConstraint', 'SubsumptionConstraint']


class Constraint(with_metaclass(abc.ABCMeta, object)):
    @staticmethod
    def from_str(string):
        if string[0] != '!':
            return MutualExclusionConstraint.from_str(string)
        else:
            return SubsumptionConstraint.from_str(string)

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def java_obj(self):
        pass


class MutualExclusionConstraint(Constraint):
    def __init__(self, labels):
        self.labels = labels

    @staticmethod
    def from_str(string):
        return MutualExclusionConstraint(string[1:].split(','))

    def __str__(self):
        return '!' + ','.join(self.labels)

    def java_obj(self):
        from jnius import autoclass
        hash_set_class = autoclass('java.util.HashSet')
        labels = hash_set_class()
        label_class = autoclass('makina.learn.classification.Label')
        for label in self.labels:
            labels.add(label_class(label))
        constraint_class = autoclass('makina.learn.classification'
                                     '.constraint.MutualExclusionConstraint')
        return constraint_class(labels)


class SubsumptionConstraint(Constraint):
    def __init__(self, parent_label, child_label):
        self.parent_label = parent_label
        self.child_label = child_label

    @staticmethod
    def from_str(string):
        string_parts = string.split(' -> ')
        return [SubsumptionConstraint(string_parts[0], child_label)
                for child_label in string_parts[1].split(',')]

    def __str__(self):
        return self.parent_label + ' -> ' + self.child_label

    def java_obj(self):
        from jnius import autoclass
        label_class = autoclass('makina.learn.classification.Label')
        constraint_class = autoclass('makina.learn.classification'
                                     '.constraint.SubsumptionConstraint')
        return constraint_class(label_class(self.parent_label),
                                label_class(self.child_label))
