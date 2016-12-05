import logging
import nig
import os
import tensorflow as tf

from functools import partial

from ..nig import mnist

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)

architectures = [[5], [1024, 512]]
use_one_hot_encoding = True
activation = nig.leaky_relu(0.01)
batch_size = 128
labeled_batch_size = 512
unlabeled_batch_size = 512
max_iter = 300
abs_loss_chg_tol = 1e-6
rel_loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 10
summary_frequency = 100
checkpoint_frequency = 10000
evaluation_frequency = 10
variable_statistics_frequency = -1
run_meta_data_frequency = -1
working_dir = os.path.join(os.getcwd(), 'working')
checkpoint_file_prefix = 'ckpt'
restore_sequentially = False
save_trained = False
optimizer = lambda: tf.train.AdamOptimizer()
gradients_processor = None  # norm_clipping(clip_norm=0.1)

# optimizer = tf.contrib.opt.ScipyOptimizerInterface
# optimizer_opts = {'options': {'maxiter': 10000}}

experiment = mnist.MNISTExperiment(
    architectures=architectures, use_one_hot_encoding=use_one_hot_encoding,
    activation=activation, batch_size=batch_size,
    labeled_batch_size=labeled_batch_size,
    unlabeled_batch_size=unlabeled_batch_size, max_iter=max_iter,
    abs_loss_chg_tol=abs_loss_chg_tol, rel_loss_chg_tol=rel_loss_chg_tol,
    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
    logging_frequency=logging_frequency, summary_frequency=summary_frequency,
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
maj_consensus_learner = partial(
    nig.ConsensusLearner, consensus_loss_weight=0.0, consensus_method='MAJ')
consensus_learner = partial(
    nig.ConsensusLearner, consensus_method='RBM', first_consensus=10,
    first_consensus_max_iter=5000, consensus_update_frequency=10,
    consensus_update_max_iter=500)

learners = [maj_consensus_learner, consensus_learner]
experiment.run(learners, show_plots=False, plots_folder=working_dir)

# test_predictions = learner.predict(
#     _get_iterator(test_data, False), ckpt=False, working_dir=working_dir,
#     ckpt_file_prefix=checkpoint_file_prefix)
# test_truth = test_data[:, -1]
# logger.info(np.mean(test_predictions == test_truth))
