import numpy as np

from . import utilities

__author__ = 'alshedivat'

SOURCE_URL = 'https://dl.dropboxusercontent.com/u/17460940/data/mediamill.npz'

TRAIN_SIZE = 35907
VALIDATION_SIZE = 3000
TEST_SIZE = 5000
NB_FEATURES = 120
NB_LABELS = 101


def maybe_download(working_dir):
    path = os.path.join(working_dir, 'mediamill/data.npz')
    if not os.path.isfile(path):
        archive_path = utilities.maybe_download('data.npz',
                                                working_dir, SOURCE_URL)


def load(working_dir):
    data_dir = os.path.join(working_dir, 'mediamill')
    # Load the data
    maybe_download(data_dir)
    data = np.load(os.path.join(data_dir, 'data.npz'))
    # Sanity checks
    assert (data['X'].shape ==
            (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, NB_FEATURES))
    assert (data['Y'].shape ==
            (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, NB_LABELS))
    # Split the data
    train_data = (data['X'][:TRAIN_SIZE],
                  data['Y'][:TRAIN_SIZE])
    val_data = (data['X'][TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE],
                data['Y'][TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE])
    test_data = (data['X'][-TEST_SIZE:],
                 data['Y'][-TEST_SIZE:])
    labels_order = data['labels_order']
    return train_data, val_data, test_data, labels_order
