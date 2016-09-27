import codecs
import numpy as np

from nig.data.utilities import deserialize_data, serialize_data
from nig.utilities.generic import raise_error

__author__ = 'eaplatanios'


def load_map(path):
    ext = path.split('.')[-1]
    if ext == 'txt':
        with codecs.open(path, 'r', encoding='UTF-8') as file:
            lines = [line.rstrip().split(' ') for line in file.readlines()]
        return {l[0]: np.array([float(v) for v in l[1:]]) for l in lines}
    elif ext == 'bin':
        return deserialize_data(path)
    raise_error(ValueError, 'Unsupported file extension %s.' % ext)


def save_map(path, mapping):
    ext = path.split('.')[-1]
    if ext == 'txt':
        with codecs.open(path, 'w', encoding='UTF-8') as file:
            for k, v in mapping.items():
                file.write(str(k) + ' ' + ' '.join(v.tolist()))
    elif ext == 'bin':
        serialize_data(mapping, path)
    else:
        raise_error(ValueError, 'Unsupported file extension %s.' % ext)
