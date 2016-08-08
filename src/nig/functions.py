import inspect

import six

__author__ = 'eaplatanios'


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


def pipeline(min_num_args=None, unique_keys=True):
    def pipeline_decorator(func):
        def func_wrapper(*args, **kwargs):
            f = PipelineFunction(func, min_num_args, args, kwargs, unique_keys)
            if f.ready():
                return f()
            return f
        return func_wrapper
    return pipeline_decorator


def _no_default_args(func):
    if six.PY2:
        spec = inspect.getargspec(func)
        if spec.defaults is not None:
            return spec.args[:-len(spec.defaults)]
        return spec.args
    return [p.name for p in inspect.signature(func).parameters.values()
            if p.default is inspect.Parameter.empty]


class PipelineFunction(object):
    """
    If min_num_args is not provided, it is set to the number of arguments of
    the provided function with no default values.

    References:
        https://mtomassoli.wordpress.com/2012/03/29/pipelining-in-python/
    """
    def __init__(self, func, min_num_args=None, args=None, kwargs=None,
                 unique_keys=True):
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
                                      if arg not in kwargs.keys()]
        if inspect.ismethod(func):
            self.__no_default_args.remove('self')
        if min_num_args is None:
            min_num_args = len(self.__no_default_args)
        self.__min_num_args = min_num_args
        self.__args = args
        self.__kwargs = kwargs
        self.__unique_keys = unique_keys
        self.__doc__ = self.__func.__call__.__doc__

    def ready(self, args=None, kwargs=None):
        if args is None:
            args = self.__args
        if kwargs is None:
            kwargs = self.__kwargs
        return self.__min_num_args <= len(args) + len(kwargs) \
            and len(self.__no_default_args) == 0

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            if args is not None:
                self.__no_default_args = self.__no_default_args[len(args):]
            if kwargs is not None:
                self.__no_default_args = [arg for arg in self.__no_default_args
                                          if arg not in kwargs.keys()]
            new_args = self.__args + args
            new_kwargs = dict.copy(self.__kwargs)
            # If unique_keys is True, we don't want repeated keyword arguments
            if self.__unique_keys and any(k in new_kwargs for k in kwargs):
                raise ValueError('Provided repeated named argument while '
                                 'unique is set to "True".')
            new_kwargs.update(kwargs)

            # Check whether it's time to evaluate the underlying function
            if self.ready(new_args, new_kwargs):
                return self.__func(*new_args, **new_kwargs)
            else:
                return PipelineFunction(self.__func, self.__min_num_args,
                                        new_args, new_kwargs,
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

        composed = PipelineFunction(composed_function, self.__min_num_args,
                                    self.__args, self.__kwargs,
                                    self.__unique_keys)
        composed.__no_default_args = self.__no_default_args
        return composed
