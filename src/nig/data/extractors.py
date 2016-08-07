from __future__ import absolute_import
from __future__ import division

import abc
import numpy as np

from nig.functions import pipeline, PipelineFunction

__author__ = 'Emmanouil Antonios Platanios'


class Extractor(PipelineFunction):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Extractor, self).__init__(self.extract, min_num_args=1)
        self.extractor = Extractor.__extractor_function(self)

    @abc.abstractmethod
    def extract(self, data):
        pass

    @staticmethod
    @pipeline(min_num_args=1)
    def __extractor_function(extractor, data):
        return extractor.extract(data)


class NPArrayColumnsExtractor(Extractor):
    def __init__(self, columns):
        super(NPArrayColumnsExtractor, self).__init__()
        self.columns = columns if isinstance(columns, list) else [columns]
        self.__extract = lambda d: d[:, self.columns]

    def extract(self, data):
        return self.__extract(data)


class PDDataFrameColumnsExtractor(Extractor):
    def __init__(self, columns):
        super(PDDataFrameColumnsExtractor, self).__init__()
        self.columns = columns
        if not isinstance(self.columns, list):
            self.__extract = lambda d: np.asarray(d[self.columns].values)
        elif isinstance(self.columns, list) and len(self.columns) == 1:
            self.__extract = \
                lambda d: np.vstack([np.asarray(value).flatten()
                                     for value in d[self.columns[0]].values])
        else:
            self.__extract = \
                lambda d: np.vstack([np.hstack([np.asarray(value).flatten()
                                                for value in values])
                                     for values in d[self.columns].values])

    def extract(self, data):
        return self.__extract(data)


class PDDataFrameTimeAggregatingExtractor(Extractor):
    def __init__(self, df, column, periods=None, frequency=None):
        super(PDDataFrameTimeAggregatingExtractor, self).__init__()
        if periods is None:
            periods = []
        self.df = df
        self.column = column
        self.periods = periods if isinstance(periods, list) else [periods]
        self.frequency = frequency

    def extract(self, data):
        df = data[self.column]
        for period in self.periods:
            df[self.column + str(period)] = \
                self.df[self.column].shift(period, self.frequency) \
                    .reindex(data.index)
        return df.values
