from __future__ import absolute_import

import csv
import numpy as np
import os
import shutil
import tempfile

from six.moves import urllib

from nig.utilities.generic import logger

__author__ = 'eaplatanios'


def load_csv(filename, target_column=-1, has_header=True):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        data_file = csv.reader(csv_file)
        if has_header:
            header = next(data_file)
            num_samples = int(header[0])
            num_features = int(header[1])
            data = np.empty([num_samples, num_features + 1])
            for i, ir in enumerate(data_file):
                data[i, -1] = np.asarray(ir.pop(target_column))
                data[i, 0:-1] = np.asarray(ir)
        else:
            data = []
            for ir in data_file:
                label = np.asarray(ir.pop(target_column))[np.newaxis]
                data.append(np.concatenate([np.asarray(ir), label], axis=0))
            data = np.array(data)
    return data


def maybe_download(filename, working_dir, source_url):
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    filepath = os.path.join(working_dir, filename)
    if not os.path.isfile(filepath):
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_filename = tmp_file.name
            urllib.request.urlretrieve(source_url, tmp_filename)
            shutil.copyfile(tmp_filename, filepath)
            size = os.path.getsize(filepath)
            logger.info('Successfully downloaded ' + filename +
                        ' (' + str(size) + ' bytes).')
    return filepath
