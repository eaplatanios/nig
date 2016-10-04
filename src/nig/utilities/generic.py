import re

from contextlib import contextmanager
from timeit import default_timer

__author__ = 'eaplatanios'


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
