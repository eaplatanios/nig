import numpy as np
import pandas as pd

from nig.data.iterators import NPArrayIterator, PDDataFrameIterator

def test_NPArrayIterator(seed=42, N=100, D=20, batch_size=8, keep_last=True):
    rng = np.random.RandomState(seed)

    # Generate dummy data and iterator
    data = rng.normal(size=(N, D))
    data_iter = NPArrayIterator(data, batch_size,
                                keep_last=keep_last)

    # Iterate, count batches, check their sizes
    num_batches = 0
    for batch in data_iter:
        num_batches += 1
        assert len(batch) <= batch_size

    expected_num_batches = len(data_iter) // batch_size
    expected_num_batches += (len(data_iter) % batch_size > 0) if keep_last \
                            else 0
    assert num_batches == expected_num_batches
