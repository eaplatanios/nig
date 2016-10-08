import numpy as np

from . import utilities

__author__ = 'alshedivat'

SOURCE_URL = 'https://dl.dropboxusercontent.com/u/17460940/data/mediamill.npz'

TRAIN_SIZE = 12920
VALIDATION_SIZE = 3000
TEST_SIZE = 3185
NB_FEATURES = 500
NB_LABELS = 983


def maybe_download(working_dir):
    path = os.path.join(working_dir, 'mediamill/data.npz')
    if not os.path.isfile(path):
        archive_path = utilities.maybe_download('data.npz',
                                                working_dir, SOURCE_URL)


def load(working_dir):
    data_dir = os.path.join(working_dir, 'mediamill')
    maybe_download(data_dir)
    data = np.load(os.path.join(data_dir, 'data.npz'))
    train_data = (data['X_train'][VALIDATION_SIZE:],
                  data['Y_train'][VALIDATION_SIZE:])
    val_data = (data['X_train'][:VALIDATION_SIZE],
                data['Y_train'][:VALIDATION_SIZE])
    test_data = (data['X_test'], data['Y_test'])
    return train_data, val_data, test_data
