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

import arff
import glob
import logging
import numpy as np
import os
import patoolib
import xml.etree.ElementTree

from . import utilities

__author__ = 'eaplatanios'

SOURCE_URL = 'http://sourceforge.net/projects/mulan/files/datasets/'

DATA_SETS = {
    'BIBTEX': 'bibtex.rar',
    'BIRDS': 'birds.rar',
    'BOOKMARKS': 'bookmarks.rar',
    'CAL500': 'CAL500.rar',
    'COREL5K': 'corel5k.rar',
    'COREL16K': 'corel16k.rar',
    'DELICIOUS': 'delicious.rar',
    'EMOTIONS': 'emotions.rar',
    'ENRON': 'enron.rar',
    # 'FLAGS': 'flags.zip',
    'GENBASE': 'genbase.rar',
    'MEDIAMILL': 'mediamill.rar',
    'MEDICAL': 'medical.rar',
    # 'NUSWIDE_CVLAD_PLUS': 'nuswide-cVLADplus.rar',
    # 'NUSWIDE_BOW': 'nuswide-bow.rar',
    'RCV1V2_SUBSET_1': 'rcv1subset1.rar',
    'RCV1V2_SUBSET_2': 'rcv1subset2.rar',
    'RCV1V2_SUBSET_3': 'rcv1subset3.rar',
    'RCV1V2_SUBSET_4': 'rcv1subset4.rar',
    'RCV1V2_SUBSET_5': 'rcv1subset5.rar',
    'SCENE': 'scene.rar',
    'TMC2007': 'tmc2007.rar',
    'YAHOO': 'yahoo.rar',
    'YEAST': 'yeast.rar'
}

logger = logging.getLogger(__name__)


def extract_data(filename):
    def separate_labels(data, labels):
        label_indices = [i[0].replace('\\\'', '\'') for i in data['attributes']]
        label_indices = [i in labels for i in label_indices]
        data_indices = [not i for i in label_indices]
        data = np.array(data['data'], dtype=np.float32)
        data = data[:, data_indices], data[:, label_indices]
        return np.column_stack(data)

    logger.info('Extracting ' + filename)
    directory = os.path.dirname(filename)
    patoolib.extract_archive(
        archive=filename, outdir=directory, interactive=False, verbosity=0)
    xml_files = glob.glob(os.path.join(directory, '*.xml'))
    if len(xml_files) > 0:
        xml_header_root = xml.etree.ElementTree.parse(xml_files[0]).getroot()
        labels = set(label.attrib['name'] for label in xml_header_root
                     if label.tag.endswith('label'))
    else:
        raise ValueError('Missing the XML header file.')
    train_data_file = glob.glob(os.path.join(directory, '*-train.arff'))[0]
    test_data_file = glob.glob(os.path.join(directory, '*-test.arff'))[0]
    train_data = arff.load(open(train_data_file, 'r'))
    test_data = arff.load(open(test_data_file, 'r'))
    train_data = separate_labels(train_data, labels)
    test_data = separate_labels(test_data, labels)
    return train_data, test_data


def load(working_dir, data_set):
    data_set = data_set.upper()
    if data_set not in DATA_SETS:
        raise ValueError('Unsupported data set name %s.' % data_set)

    working_dir = os.path.join(working_dir, data_set.lower())
    train_data_file = os.path.join(working_dir, 'train_data.npy')
    test_data_file = os.path.join(working_dir, 'test_data.npy')
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
        filename = DATA_SETS[data_set]
        local_file = utilities.maybe_download(
            filename=filename, working_dir=working_dir,
            source_url=SOURCE_URL + filename)
        train_data, test_data = extract_data(local_file)
        np.save(train_data_file, train_data)
        np.save(test_data_file, test_data)
    else:
        train_data = np.load(train_data_file)
        test_data = np.load(test_data_file)
    return train_data, test_data
