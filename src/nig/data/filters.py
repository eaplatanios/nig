from __future__ import absolute_import
from __future__ import division

import abc

from nig.functions import PipelineFunction, pipeline

__author__ = 'eaplatanios'


class Filter(PipelineFunction):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Filter, self).__init__(self.filter, min_num_args=1)
        self.extractor = Filter.__filter_function(self)

    @abc.abstractmethod
    def filter(self, data):
        pass

    @staticmethod
    @pipeline(min_num_args=2)
    def __filter_function(filter, data):
        return filter.filter(data)


class PDDataFrameAnyNaNFilter(Filter):
    def __init__(self, columns):
        super(PDDataFrameAnyNaNFilter, self).__init__()
        self.columns = columns if isinstance(columns, list) else [columns]
        self.__filter = lambda d: d[:, self.columns]

    def filter(self, data):
        return data[self.columns].isnull().values.any()
