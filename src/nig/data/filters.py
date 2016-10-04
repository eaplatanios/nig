import abc
import numpy as np

from six import with_metaclass

from nig.utilities.functions import PipelineFunction

__author__ = 'eaplatanios'


class Filter(with_metaclass(abc.ABCMeta, PipelineFunction)):
    """A :class:`Filter` receives some data as input and provides a function
    for filtering that data.

    The filtering is performed along samples (i.e., some samples are being
    filtered out).

    Note:
        The returned data can contain fewer samples than the provided data.
    """

    def __init__(self):
        super(Filter, self).__init__(self.filter)

    @abc.abstractmethod
    def filter(self, data):
        pass


class NPFilter(with_metaclass(abc.ABCMeta, Filter)):
    """A :class:`NPFilter` receives a :class:`np.ndarray` as input and
    provides a function for filtering it across its first dimension.
    """
    def __init__(self):
        super(NPFilter, self).__init__()

    @abc.abstractmethod
    def filter(self, data):
        pass


class TFFilterMixin(with_metaclass(abc.ABCMeta)):
    """A :class:`TFProcessorMixin` is supposed to be used mixed in  with the
    :class:`NPProcessor` abstract class, whenever a TensorFlow op can be defined
    for the corresponding transform, operating on a :class:`tf.Tensor`. This
    allows for the corresponding transform to be computed by TensorFlow.

    Note:
        The first dimension of the input tensor is always preserved as it is
        meant to correspond to an index over samples. Processors are required
        not to alter that index.
    """
    @abc.abstractmethod
    def filter_op(self, tensor):  # TODO: Somehow make this pipelinable.
        pass


class NPAnyNaNFilter(NPFilter):
    def __init__(self, columns):
        super(NPAnyNaNFilter, self).__init__()
        self.columns = columns if isinstance(columns, list) else [columns]

    def filter(self, data):
        return data[~np.isnan(data).any(axis=1)]

    # TODO: Add TensorFlow mixin support.
    # def filter_op(self, tensor):
    #     tf.boolean_mask(tensor, tf.is_finite(tensor))
    #     pass


class PDFilter(with_metaclass(abc.ABCMeta, Filter)):
    """A :class:`PDFilter` receives a :class:`pd.Series` or a
    :class:`pd.DataFrame` as input and provides a function for filtering it
    across its index.
    """
    def __init__(self):
        super(PDFilter, self).__init__()

    @abc.abstractmethod
    def filter(self, data):
        pass


class PDDataFrameAnyNaNFilter(PDFilter):
    def __init__(self, columns):
        super(PDDataFrameAnyNaNFilter, self).__init__()
        self.columns = columns if isinstance(columns, list) else [columns]

    def filter(self, data):
        return data[self.columns].isnull().values.any()
