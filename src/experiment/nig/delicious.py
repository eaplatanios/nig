import logging
import nig
import os
import tensorflow as tf

from collections import OrderedDict
from functools import partial

from experiment.nig import experiments

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)

seed = 9999
architectures = [[16], [256],
                 [16, 16], [256, 256],
                 [16, 16, 16, 16], [256, 256, 256, 256],
                 [512, 256], [1024, 512, 256], [16, 16, 16, 16]]
use_one_hot_encoding = True
activation = nig.leaky_relu(0.01)
labeled_batch_size = 128
unlabeled_batch_size = 128
test_data_proportion = 0.95
max_iter = 10000
abs_loss_chg_tol = 1e-6
rel_loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 10
summary_frequency = -1
checkpoint_frequency = -1
evaluation_frequency = 50
variable_statistics_frequency = -1
run_meta_data_frequency = -1
working_dir = os.path.join(os.getcwd(), 'working', 'delicious')
checkpoint_file_prefix = 'ckpt'
restore_sequentially = False
save_trained = False
optimizer = lambda: tf.train.AdagradOptimizer(0.1)  # nig.gradient_descent(1e-1, decay_rate=0.99)
gradients_processor = None  # lambda g: tf.clip_by_norm(g, 1e1)

# optimizer = tf.contrib.opt.ScipyOptimizerInterface
# optimizer_opts = {'options': {'maxiter': 10000}}

# def consensus_loss_metric(outputs, consensus):
#     with tf.name_scope('consensus_loss_metric'):
#         outputs = tf.exp(outputs)
#         metric = tf.square(tf.sub(outputs, consensus))
#         metric = tf.reduce_sum(metric)
#     return metric
consensus_loss_metric = None

consensus_configurations = [
    ('Majority 0.0', {'consensus_method': nig.Vote(
        trainable=False, hard_vote=False, argmax_vote=False),
                      'consensus_loss_weight': 0.0,
                      'consensus_loss_metric': None,
                      'first_consensus': 10}),
    # ('Majority 0.1', {'consensus_method': nig.Vote(trainable=False),
    #                   'consensus_loss_weight': 0.1,
    #                   'consensus_loss_metric': None,
    #                   'first_consensus': 10}),
    # ('Majority 0.5', {'consensus_method': nig.Vote(
    #     trainable=False, hard_vote=False, argmax_vote=False),
    #                   'consensus_loss_weight': 0.5,
    #                   'consensus_loss_metric': None,
    #                   'first_consensus': 10}),
    ('Majority 1.0', {'consensus_method': nig.Vote(
        trainable=False, hard_vote=False, argmax_vote=False),
                      'consensus_loss_weight': 1.0,
                      'consensus_loss_metric': None,
                      'first_consensus': 10}),
    ('Trainable Majority 0.0', {'consensus_method': nig.Vote(
        trainable=True, hard_vote=False, argmax_vote=False),
                                'consensus_loss_weight': 0.0,
                                'consensus_loss_metric': None,
                                'first_consensus': 10,
                                'first_consensus_max_iter': 1000,
                                'consensus_update_frequency': 10,
                                'consensus_update_max_iter': 500}),
    # ('Trainable Majority 0.1', {'consensus_method': nig.Vote(
    #     trainable=True, hard_vote=False, argmax_vote=False),
    #                             'consensus_loss_weight': 0.1,
    #                             'consensus_loss_metric': None,
    #                             'first_consensus': 10,
    #                             'first_consensus_max_iter': 1000,
    #                             'consensus_update_frequency': 10,
    #                             'consensus_update_max_iter': 500}),
    # ('Trainable Majority 0.5', {'consensus_method': nig.Vote(
    #     trainable=True, hard_vote=False, argmax_vote=False),
    #                             'consensus_loss_weight': 0.5,
    #                             'consensus_loss_metric': None,
    #                             'first_consensus': 10,
    #                             'first_consensus_max_iter': 1000,
    #                             'consensus_update_frequency': 10,
    #                             'consensus_update_max_iter': 500}),
    ('Trainable Majority 1.0', {'consensus_method': nig.Vote(
        trainable=True, hard_vote=False, argmax_vote=False),
                                'consensus_loss_weight': 1.0,
                                'consensus_loss_metric': None,
                                'first_consensus': 10,
                                'first_consensus_max_iter': 1000,
                                'consensus_update_frequency': 10,
                                'consensus_update_max_iter': 500}),
    # ('Hard Majority 0.0', {'consensus_method': nig.HardMajorityVote(),
    #                        'consensus_loss_weight': 0.0,
    #                        'consensus_loss_metric': None,
    #                        'first_consensus': 10}),
    # ('Hard Majority 0.1', {'consensus_method': nig.HardMajorityVote(),
    #                        'consensus_loss_weight': 0.1,
    #                        'consensus_loss_metric': None,
    #                        'first_consensus': 10}),
    # ('Hard Majority 0.5', {'consensus_method': nig.HardMajorityVote(),
    #                        'consensus_loss_weight': 0.5,
    #                        'consensus_loss_metric': None,
    #                        'first_consensus': 10}),
    # ('Hard Majority 1.0', {'consensus_method': nig.HardMajorityVote(),
    #                        'consensus_loss_weight': 1.0,
    #                        'consensus_loss_metric': None,
    #                        'first_consensus': 10}),
    # ('RBM 0.0', {'consensus_method': nig.RBMConsensus(),
    #              'consensus_loss_weight': 0.0,
    #              'consensus_loss_metric': None,
    #              'first_consensus': 10,
    #              'first_consensus_max_iter': 10000,
    #              'consensus_update_frequency': 10,
    #              'consensus_update_max_iter': 500}),
    # ('RBM 0.1', {'consensus_method': nig.RBMConsensus(),
    #              'consensus_loss_weight': 0.1,
    #              'consensus_loss_metric': None,
    #              'first_consensus': 10,
    #              'first_consensus_max_iter': 10000,
    #              'consensus_update_frequency': 10,
    #              'consensus_update_max_iter': 500}),
    # ('RBM 0.5', {'consensus_method': nig.RBMConsensus(),
    #              'consensus_loss_weight': 0.5,
    #              'consensus_loss_metric': None,
    #              'first_consensus': 10,
    #              'first_consensus_max_iter': 10000,
    #              'consensus_update_frequency': 10,
    #              'consensus_update_max_iter': 500}),
    # ('RBM 1.0', {'consensus_method': nig.RBMConsensus(),
    #              'consensus_loss_weight': 1.0,
    #              'consensus_loss_metric': None,
    #              'first_consensus': 10,
    #              'first_consensus_max_iter': 10000,
    #              'consensus_update_frequency': 10,
    #              'consensus_update_max_iter': 500}),
]

with nig.dummy():  # tf.device('/cpu:0'):
    experiment = experiments.DeliciousExperiment(
        architectures=architectures, activation=activation,
        labeled_batch_size=labeled_batch_size,
        unlabeled_batch_size=unlabeled_batch_size,
        test_data_proportion=test_data_proportion, max_iter=max_iter,
        abs_loss_chg_tol=abs_loss_chg_tol, rel_loss_chg_tol=rel_loss_chg_tol,
        loss_chg_iter_below_tol=loss_chg_iter_below_tol,
        logging_frequency=logging_frequency,
        summary_frequency=summary_frequency,
        checkpoint_frequency=checkpoint_frequency,
        evaluation_frequency=evaluation_frequency,
        variable_statistics_frequency=variable_statistics_frequency,
        run_meta_data_frequency=run_meta_data_frequency,
        working_dir=working_dir, checkpoint_file_prefix=checkpoint_file_prefix,
        restore_sequentially=restore_sequentially, save_trained=save_trained,
        optimizer=optimizer, gradients_processor=gradients_processor)
    learners = []
    for name, configuration in consensus_configurations:
        learner = partial(nig.ConsensusLearner, **configuration)
        learners.append((name, learner))
    learners = OrderedDict(learners)
    results = experiment.run(learners)
experiments.save_results(
    results, filename=os.path.join(os.getcwd(), 'working', 'results.pk'),
    update=True, use_backup=True, delete_backup=False)
# results = experiments.load_results(
#     filename=os.path.join(working_dir, 'results.pk'))
experiments.plot_results(results)
