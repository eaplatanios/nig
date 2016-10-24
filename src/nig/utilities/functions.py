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

import inspect
import logging
import six

from functools import wraps

from .generic import elapsed_timer

__author__ = 'eaplatanios'

__all__ = ['identity', 'compose', 'pipe', 'memoize', 'time', 'pipeline',
           'PipelineFunction']

logger = logging.getLogger(__name__)


def identity(arg):
    return arg


def compose(*functions):
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


def pipe(*functions):
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner


def memoize(func):
    @wraps(func)
    def memoized(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        result = memoized._cache.get(key, None)
        if result is None:
            result = func(*args, **kwargs)
            memoized._cache[key] = result
        return result
    memoized._cache = {}
    return memoized


def time(func):
    @wraps(func)
    def timed(*args, **kwargs):
        with elapsed_timer() as elapsed:
            result = func(*args, **kwargs)
        logger.info(func.__name__ + ' took %.6f seconds.' % elapsed())
        return result
    return timed


def pipeline(min_num_args=None, unique_keys=True):
    def pipeline_decorator(func):
        return PipelineFunction(func,
                                min_num_args=min_num_args,
                                unique_keys=unique_keys)
    return pipeline_decorator


def _no_default_args(func):
    if six.PY2:
        spec = inspect.getargspec(func)
        args = spec.args
        if inspect.ismethod(func):
            args.remove('self')
        if spec.defaults is not None:
            return args[:-len(spec.defaults)]
        return args
    return [p.name for p in inspect.signature(func).parameters.values()
            if p.default is inspect.Parameter.empty]


class PipelineFunction(object):
    """
    If min_num_args is not provided, it is set to the number of arguments of
    the provided function with no default values.

    References:
        https://mtomassoli.wordpress.com/2012/03/29/pipelining-in-python/
    """
    def __init__(self, func, args=None, kwargs=None,
                 min_num_args=None, unique_keys=True):
        if args is None:
            args = tuple()
        elif not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        self.__func = func
        self.__no_default_args = _no_default_args(func)[len(args):]
        if kwargs is not None:
            self.__no_default_args = [arg for arg in self.__no_default_args
                                      if arg not in kwargs]
        if min_num_args is None:
            min_num_args = len(self.__no_default_args)
        self.__min_num_args = min_num_args
        self.__args = args
        self.__kwargs = kwargs
        self.__unique_keys = unique_keys
        self.__doc__ = self.__func.__call__.__doc__

    def ready(self, args=None, kwargs=None, no_default_args=None):
        if args is None:
            args = self.__args
        if kwargs is None:
            kwargs = self.__kwargs
        if no_default_args is None:
            no_default_args = self.__no_default_args
        return (self.__min_num_args <= len(args) + len(kwargs) and
                len(no_default_args) == 0)

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            no_default_args = self.__no_default_args[:]
            if args is not None:
                no_default_args = no_default_args[len(args):]
            if kwargs is not None:
                no_default_args = [arg for arg in no_default_args
                                   if arg not in kwargs]
            new_args = self.__args + args
            new_kwargs = dict.copy(self.__kwargs)
            # If unique_keys is True, we don't want repeated keyword arguments
            if self.__unique_keys and any(k in new_kwargs for k in kwargs):
                raise ValueError('Provided repeated named argument while '
                                 'unique is set to `True`.')
            new_kwargs.update(kwargs)

            # Check whether it's time to evaluate the underlying function
            if self.ready(new_args, new_kwargs, no_default_args):
                return self.__func(*new_args, **new_kwargs)
            else:
                return PipelineFunction(self.__func, new_args, new_kwargs,
                                        self.__min_num_args,
                                        self.__unique_keys)
        else:  # If no more arguments are passed in, evaluation is forced
            return self.__func(*self.__args, **self.__kwargs)

    def __rrshift__(self, arg):
        """Forces evaluation of the pipeline function using the provided
        argument."""
        return self.__func(*(self.__args + (arg,)), **self.__kwargs)

    def __or__(self, other):
        """Composes the pipeline function with another pipeline function."""
        if not isinstance(other, PipelineFunction):
            raise TypeError('A PipelineFunction can only be composed with '
                            'another PipelineFunction.')

        def composed_function(*args, **kwargs):
            result = self.__func(*args, **kwargs)
            return other.__func(*(other.__args + (result,)), **other.__kwargs)

        composed = PipelineFunction(composed_function,
                                    self.__args, self.__kwargs,
                                    self.__min_num_args, self.__unique_keys)
        composed.__no_default_args = self.__no_default_args
        return composed
