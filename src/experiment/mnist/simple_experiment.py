import logging
import nig
import os
import tensorflow as tf

from collections import OrderedDict
from functools import partial

from ..nig import mnist

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)

architectures = [[1], [16], [128], [128, 64, 32], [1024, 512, 256], [2048]]
use_one_hot_encoding = True
activation = nig.leaky_relu(0.01)
labeled_batch_size = 128
unlabeled_batch_size = 1024
test_data_proportion = 0.95
max_iter = 300
abs_loss_chg_tol = 1e-6
rel_loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 20
summary_frequency = -1
checkpoint_frequency = -1
evaluation_frequency = 20
variable_statistics_frequency = -1
run_meta_data_frequency = -1
working_dir = os.path.join(os.getcwd(), 'working', 'mnist')
checkpoint_file_prefix = 'ckpt'
restore_sequentially = False
save_trained = False
optimizer = lambda: tf.train.GradientDescentOptimizer(0.1)
gradients_processor = None  # norm_clipping(clip_norm=0.1)

# optimizer = tf.contrib.opt.ScipyOptimizerInterface
# optimizer_opts = {'options': {'maxiter': 10000}}


# def consensus_loss_metric(outputs, consensus):
#     with tf.name_scope('consensus_loss_metric'):
#         outputs = tf.exp(outputs)
#         metric = tf.square(tf.sub(outputs, consensus))
#         metric = tf.reduce_sum(metric)
#     return metric
consensus_loss_metric = None

with tf.device('/cpu:0'):
    experiment = mnist.MNISTExperiment(
        architectures=architectures, use_one_hot_encoding=use_one_hot_encoding,
        activation=activation, labeled_batch_size=labeled_batch_size,
        unlabeled_batch_size=unlabeled_batch_size,
        test_data_proportion=test_data_proportion, max_iter=max_iter,
        abs_loss_chg_tol=abs_loss_chg_tol, rel_loss_chg_tol=rel_loss_chg_tol,
        loss_chg_iter_below_tol=loss_chg_iter_below_tol,
        logging_frequency=logging_frequency,
        summary_frequency=summary_frequency,
        checkpoint_frequency=checkpoint_frequency,
        evaluation_frequency=evaluation_frequency,
        variable_statistics_frequency=variable_statistics_frequency,
        run_meta_data_frequency=run_meta_data_frequency, working_dir=working_dir,
        checkpoint_file_prefix=checkpoint_file_prefix,
        restore_sequentially=restore_sequentially, save_trained=save_trained,
        optimizer=optimizer, gradients_processor=gradients_processor)
    # maj_trust_based_learner = partial(nig.TrustBasedLearner, first_trust_update=max_iter + 1)
    # trust_based_learner = partial(
    #     nig.TrustBasedLearner, first_trust_update=10, trust_update_frequency=10)
    maj_00_consensus_learner = partial(
        nig.ConsensusLearner, consensus_loss_weight=0.0, consensus_method='MAJ',
        consensus_loss_metric=consensus_loss_metric, first_consensus=10)
    maj_0_consensus_learner = partial(
        nig.ConsensusLearner, consensus_loss_weight=1e0, consensus_method='MAJ',
        consensus_loss_metric=consensus_loss_metric, first_consensus=10)
    # maj_1_consensus_learner = partial(
    #     nig.ConsensusLearner, consensus_loss_weight=1e1, consensus_method='MAJ',
    #     consensus_loss_metric=consensus_loss_metric, first_consensus=10)
    # hmaj_00_consensus_learner = partial(
    #     nig.ConsensusLearner, consensus_loss_weight=0.0,
    #     consensus_method='HMAJ', consensus_loss_metric=consensus_loss_metric,
    #     first_consensus=10)
    # hmaj_0_consensus_learner = partial(
    #     nig.ConsensusLearner, consensus_loss_weight=1e0,
    #     consensus_method='HMAJ', consensus_loss_metric=consensus_loss_metric,
    #     first_consensus=10)
    # hmaj_1_consensus_learner = partial(
    #     nig.ConsensusLearner, consensus_loss_weight=1e1,
    #     consensus_method='HMAJ', consensus_loss_metric=consensus_loss_metric,
    #     first_consensus=10)
    consensus_00_learner = partial(
        nig.ConsensusLearner, consensus_loss_weight=0.0, consensus_method='RBM',
        consensus_loss_metric=consensus_loss_metric, first_consensus=10,
        first_consensus_max_iter=5000, consensus_update_frequency=10,
        consensus_update_max_iter=500)
    consensus_0_learner = partial(
        nig.ConsensusLearner, consensus_loss_weight=1e0, consensus_method='RBM',
        consensus_loss_metric=consensus_loss_metric, first_consensus=10,
        first_consensus_max_iter=5000, consensus_update_frequency=10,
        consensus_update_max_iter=500)

    learners = OrderedDict([#('Majority-0.0', maj_00_consensus_learner),
                            #('Majority-1.0', maj_0_consensus_learner),
                            # ('Majority-10.0', maj_1_consensus_learner)])
                            # ('Hard Majority-0.0', hmaj_00_consensus_learner),
                            # ('Hard Majority-1.0', hmaj_0_consensus_learner),
                            # ('Hard Majority-10.0', hmaj_1_consensus_learner)])
                            ('RBM-0.0', consensus_00_learner),
                            ('RBM-1.0', consensus_0_learner)])
    experiment.run(learners, show_plots=True, plots_folder=None)

    # test_predictions = learner.predict(
    #     _get_iterator(test_data, False), ckpt=False, working_dir=working_dir,
    #     ckpt_file_prefix=checkpoint_file_prefix)
    # test_truth = test_data[:, -1]
    # logger.info(np.mean(test_predictions == test_truth))
