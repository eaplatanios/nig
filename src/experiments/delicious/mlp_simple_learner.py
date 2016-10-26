import logging
import tensorflow as tf

from nig.data.loaders import delicious
from nig.data.processors import *
from nig.learning.callbacks import *
from nig.learning.metrics import *
from nig.learning.learners import *
from nig.learning.optimizers import *
from nig.learning.processors import norm_summary, norm_clipping
from nig.models.adios import MultiLabelMLP
from nig.utilities.experiment import *

__author__ = 'alshedivat'

logger = logging.getLogger(__name__)


def main():
    # Load the data
    train_data, val_data, test_data, _ = delicious.load('data')

    # Construct the model
    architectures = [[5], [16, 32, 16]]
    optimizer = lambda: gradient_descent(1e-1, decay_rate=0.99,
                                         learning_rate_summary=True)
    optimizer_opts = {'batch_size': 100,
                      'max_iter': 1000,
                      'abs_loss_chg_tol': 1e-10,
                      'rel_loss_chg_tol': 1e-6,
                      'loss_chg_iter_below_tol': 5,
                      'grads_processor': None}

    loss = CrossEntropyOneHotEncodingMetric()
    eval_metric = HammingLossMetric()

    models = [MultiLabelMLP(
        train_data[0].shape[1], train_data[1].shape[1], architecture,
        activation=tf.nn.relu, loss=loss, loss_summary=True,
        optimizer=optimizer, optimizer_opts=optimizer_opts)
              for architecture in architectures]

    # Fit the model
    learner = train(models, train_data, learner='ValidationSetLearner',
                    validation_data=val_data, eval_metric=eval_metric)

    # Test the model
    # TODO

    # Save results
    # TODO

if __name__ == '__main__':
    main()
