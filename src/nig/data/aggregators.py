from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np

from nig.functions import pipeline

__author__ = 'Emmanouil Antonios Platanios'


class Aggregator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.aggregator = Aggregator.__aggregator_function(self)

    @abc.abstractmethod
    def aggregate(self, data):
        pass

    @staticmethod
    @pipeline(min_num_args=1)
    def __aggregator_function(aggregator, data):
        return aggregator.aggregate(data)


class NPArrayColumnsAggregator(Aggregator):
    def __init__(self, columns=None):
        super(NPArrayColumnsAggregator, self).__init__()
        if columns is None:
            self.__aggregate = lambda d: np.column_stack(d)
        else:
            self.__aggregate = lambda d: np.column_stack([d[col]
                                                          for col in columns])

    def aggregate(self, data):
        return self.__aggregate(data)
