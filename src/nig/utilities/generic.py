from __future__ import absolute_import

import inspect
import logging
import re
import sys

from contextlib import contextmanager
from timeit import default_timer

__author__ = 'eaplatanios'

logging.basicConfig(
    level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)-15s - %(levelname)-7s - %(name)-20s - %(message)s')
logger = logging.getLogger(inspect.currentframe().f_back.f_globals['__name__'])


def add_logging_file_handler(path):
    logger.addHandler(logging.FileHandler(path))


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
