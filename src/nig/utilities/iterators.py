import abc
from six import with_metaclass

__author__ = 'eaplatanios'


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
    def __init__(self, iterators):
        self._iterators = iterators

    def next(self):
        return tuple([i.next() for i in self._iterators])

    def reset(self):
        for i in self._iterators:
            i.reset()

    def reset_copy(self):
        return ZipIterator([i.reset_copy() for i in self._iterators])

    def __len__(self):
        raise NotImplementedError

    def remaining_length(self):
        raise NotImplementedError
