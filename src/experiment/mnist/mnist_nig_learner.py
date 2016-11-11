import logging
import nig
import numpy as np
import os
import tensorflow as tf

from nig.data.loaders import mnist

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)

use_one_hot_encoding = True
architectures = [[5], [16, 32, 16], [128, 64, 32], [256, 128, 64, 32]]
activation = tf.nn.relu
batch_size = 100
labeled_batch_size = 1000
unlabeled_batch_size = 1000
max_iter = 10000
abs_loss_chg_tol = 1e-10
rel_loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 10
summary_frequency = 100
checkpoint_frequency = 1000
evaluation_frequency = 100
working_dir = os.path.join(os.getcwd(), 'run')
checkpoint_file_prefix = 'ckpt'
restore_sequentially = False
save_trained = True
gradients_processor = None #norm_clipping(clip_norm=0.1) \
#| norm_summary(name='gradients/norm')
optimizer = lambda: tf.train.AdamOptimizer()
optimizer_opts = {'batch_size': batch_size,
                  'max_iter': max_iter,
                  'abs_loss_chg_tol': abs_loss_chg_tol,
                  'rel_loss_chg_tol': rel_loss_chg_tol,
                  'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
                  'grads_processor': gradients_processor}
# optimizer = tf.contrib.opt.ScipyOptimizerInterface
# optimizer_opts = {'options': {'maxiter': 10000}}

train_data, val_data, test_data = mnist.load('data', float_images=True)

inputs_pipeline = nig.ColumnsExtractor(list(range(784)))
labels_pipeline = nig.ColumnsExtractor(784)
if use_one_hot_encoding:
    labels_pipeline = labels_pipeline | \
                      nig.DataTypeEncoder(np.int8) | \
                      nig.OneHotEncoder(10)


def _get_iterator(mnist_data, include_labels=True):
    pipelines = [inputs_pipeline]
    if include_labels:
        pipelines.append(labels_pipeline)
    return nig.NPArrayIterator(
        mnist_data, batch_size, shuffle=False, cycle=False, cycle_shuffle=False,
        keep_last=True, pipelines=pipelines)

loss = nig.CrossEntropyOneHotEncodingMetric() if use_one_hot_encoding \
    else nig.CrossEntropyIntegerEncodingMetric()
eval_metric = nig.AccuracyOneHotEncodingMetric() if use_one_hot_encoding \
    else nig.AccuracyIntegerEncodingMetric()

models = [nig.MultiLayerPerceptron(
    784, 10, architecture, activation=activation,
    softmax_output=use_one_hot_encoding, use_log=use_one_hot_encoding,
    train_outputs_one_hot=use_one_hot_encoding, loss=loss, loss_summary=True,
    optimizer=optimizer, optimizer_opts=optimizer_opts)
          for architecture in architectures]

callbacks = [
    nig.LoggerCallback(frequency=logging_frequency),
    nig.SummaryWriterCallback(frequency=summary_frequency),
    nig.RunMetaDataSummaryWriterCallback(
        frequency=1000, trace_level=tf.RunOptions.FULL_TRACE),
    nig.VariableStatisticsSummaryWriterCallback(
        frequency=200, variables='trainable'),
    nig.CheckpointWriterCallback(
        frequency=checkpoint_frequency, file_prefix=checkpoint_file_prefix),
    nig.EvaluationCallback(
        frequency=evaluation_frequency, data=_get_iterator(train_data),
        metrics=eval_metric, name='eval/train'),
    nig.EvaluationCallback(
        frequency=evaluation_frequency, data=_get_iterator(val_data),
        metrics=eval_metric, name='eval/val'),
    nig.EvaluationCallback(
        frequency=evaluation_frequency, data=_get_iterator(test_data),
        metrics=eval_metric, name='eval/test')]

# learner = SimpleLearner(
#     model=models[0], predict_postprocess=lambda l: tf.argmax(l, 1))
learner = nig.TrustBasedLearner(
    models=models, predict_postprocess=lambda l: tf.argmax(l, 1))

labeled_data = nig.get_iterator(
    data=np.concatenate([train_data, val_data], axis=0),
    batch_size=labeled_batch_size, shuffle=True, cycle=True, cycle_shuffle=True,
    pipelines=[inputs_pipeline, labels_pipeline])
unlabeled_data = nig.get_iterator(
    data=test_data, batch_size=unlabeled_batch_size, shuffle=True, cycle=True,
    cycle_shuffle=True, pipelines=[inputs_pipeline])

learner.train(
    data=train_data, pipelines=[inputs_pipeline, labels_pipeline],
    init_option=True, per_model_callbacks=None,
    combined_model_callbacks=callbacks, working_dir=working_dir,
    ckpt_file_prefix=checkpoint_file_prefix,
    restore_sequentially=restore_sequentially, save_trained=save_trained,
    labeled_data=labeled_data, unlabeled_data=unlabeled_data)
test_predictions = learner.predict(
    _get_iterator(test_data, False), ckpt=-1, working_dir=working_dir,
    ckpt_file_prefix=checkpoint_file_prefix)
test_truth = test_data[:, -1]
logger.info(np.mean(test_predictions == test_truth))
