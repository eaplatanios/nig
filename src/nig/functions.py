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


def pipeline(unique=True, min_num_args=None):
    def pipeline_decorator(func):
        def func_wrapper(*args, **kwargs):
            return PipelineFunction(func, min_num_args, args, kwargs, unique)
        return func_wrapper
    return pipeline_decorator


class PipelineFunction:
    """
    If min_num_args is not provided, it is taken to be max(0, x - 1) where
        x is the number of non-defaulted arguments to the function. If the
        function has *args, then min_num_args defaults to the number of
        explicit arguments taken by the function.

    References:
        https://mtomassoli.wordpress.com/2012/03/29/pipelining-in-python/
    """
    def __init__(self, func, min_num_args, args=(), kwargs=None,
                 unique_keys=True):
        if kwargs is None:
            kwargs = dict()
        self.__func = func
        self.__min_num_args = min_num_args
        self.__args = args
        self.__kwargs = kwargs
        self.__unique_keys = unique_keys
        self.__doc__ = self.__func.__call__.__doc__

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            new_args = self.__args + args
            new_kwargs = dict.copy(self.__kwargs)
            # If unique is True, we don't want repeated keyword arguments
            if self.__unique_keys and any(k in new_kwargs for k in kwargs):
                raise ValueError('Provided repeated named argument while '
                                 'unique is set to "True".')
            new_kwargs.update(kwargs)

            # Check whether it's time to evaluate the underlying function
            if self.__min_num_args is not None \
                    and self.__min_num_args <= len(new_args) + len(new_kwargs):
                return self.__func(*new_args, **new_kwargs)
            else:
                return PipelineFunction(self.__func, new_args, new_kwargs,
                                        self.__unique_keys, self.__min_num_args)
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

        return PipelineFunction(composed_function, self.__args, self.__kwargs,
                                self.__unique_keys, self.__min_num_args)
