import re
import six

from contextlib import contextmanager
from timeit import default_timer

__author__ = ['eaplatanios', 'alshedivat']


@contextmanager
def dummy():
    yield None


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapsed = lambda: default_timer() - start
    yield lambda: elapsed()
    end = default_timer()
    elapsed = lambda: end - start


def escape_glob(path):
    characters = ['[', ']', '?', '!']
    replacements = {re.escape(char): '[' + char + ']' for char in characters}
    pattern = re.compile('|'.join(replacements.keys()))
    return pattern.sub(lambda m: replacements[re.escape(m.group(0))], path)


def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    """The function is stolen from keras.utils.generic_utils.
    """
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif type(identifier) is dict:
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
    return identifier
