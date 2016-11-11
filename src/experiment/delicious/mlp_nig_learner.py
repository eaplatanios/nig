import logging
import nig

from nig.data.loaders import delicious

from experiment.simple_experiment import *

__author__ = 'alshedivat'

logger = logging.getLogger(__name__)


def main():
    # Load the data
    train_data, val_data, test_data, _ = delicious.load('data')

    # Construct the model
    architectures = [[512], [1024], [512, 512], [1024, 1024]]
    optimizer = lambda: nig.gradient_descent(1e-1, decay_rate=0.99,
                                             learning_rate_summary=True)
    optimizer_opts = {'batch_size': 2**7,  # good to have powers of 2
                      'max_iter': 1000,
                      'abs_loss_chg_tol': 1e-10,
                      'rel_loss_chg_tol': 1e-6,
                      'loss_chg_iter_below_tol': 5,
                      'grads_processor': None}

    loss = nig.CrossEntropyOneHotEncodingLogitsMetric()
    eval_metric = nig.HammingLossMetric()

    models = [nig.MultiLabelMLP(
        train_data[0].shape[1], train_data[1].shape[1], architecture,
        activation=tf.nn.relu, loss=loss, loss_summary=True,
        optimizer=optimizer, optimizer_opts=optimizer_opts)
              for architecture in architectures]

    # Fit the model
    learner = train(models, train_data, learner=nig.NIGLearner,
                    validation_data=val_data, eval_metric=eval_metric)

    # Test the model
    # TODO

    # Save results
    # TODO

if __name__ == '__main__':
    main()
