from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import numpy as np
import os
import tensorflow as tf

from nig.data.aggregators import NPArrayColumnsAggregator
from nig.data.encoders import OneHotEncoder, DataTypeEncoder
from nig.data.extractors import NPArrayColumnsExtractor
from nig.data.iterators import NPArrayIterator
from nig.learn.callbacks import LoggerCallback, SummaryWriterCallback, \
    CheckpointWriterCallback, EvaluationCallback
from nig.learn.metrics import CrossEntropyOneHotEncodingMetric, \
    AccuracyOneHotEncodingMetric, CrossEntropyIntegerEncodingMetric, \
    AccuracyIntegerEncodingMetric
from nig.learn.learners import SimpleLearner
from nig.learn.symbols import MultiLayerPerceptron

use_one_hot_encoding = False
architecture = [128, 32]
activation = tf.nn.relu
optimizer = tf.train.GradientDescentOptimizer(1e-2)
batch_size = 100
max_iter = 100000
loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 100
summary_frequency = 100
checkpoint_frequency = 1000
evaluation_frequency = 1000
working_dir = os.getcwd()
checkpoint_file_prefix = 'checkpoint'
restore_sequentially = False
save_trained = False

data = read_data_sets('data', False)

aggregate = NPArrayColumnsAggregator().aggregator
inputs_pipeline = NPArrayColumnsExtractor(list(range(784))).extractor
labels_pipeline = NPArrayColumnsExtractor(784).extractor
if use_one_hot_encoding:
    labels_pipeline = labels_pipeline | DataTypeEncoder(np.int8).encoder | \
                      OneHotEncoder(10).encoder


def get_tf_mnist_data(tf_mnist_dataset):
    return tf_mnist_dataset.next_batch(tf_mnist_dataset.num_examples, False)


def get_iterator(tf_mnist_data, include_labels=True):
    pipelines = [inputs_pipeline]
    if include_labels:
        pipelines.append(labels_pipeline)
    return NPArrayIterator(
        tf_mnist_data >> aggregate, batch_size, shuffle=False, cycle=False,
        cycle_shuffle=False, keep_last_batch=True, pipelines=pipelines)

symbol = MultiLayerPerceptron(784, 10, architecture, activation=activation,
                              softmax_output=use_one_hot_encoding,
                              use_log=use_one_hot_encoding)

outputs_dtype = tf.float32 if use_one_hot_encoding else tf.int32
output_shape = 10 if use_one_hot_encoding else 1

learner = SimpleLearner(symbol, inputs_dtype=tf.float32,
                        outputs_dtype=outputs_dtype,
                        output_shape=output_shape, loss_summary=True,
                        gradient_norm_summary=False,
                        predict_postprocess=lambda l: tf.argmax(l, 1))

loss = CrossEntropyOneHotEncodingMetric() if use_one_hot_encoding \
    else CrossEntropyIntegerEncodingMetric()
eval_metric = AccuracyOneHotEncodingMetric() if use_one_hot_encoding \
    else AccuracyIntegerEncodingMetric()

callbacks = []
callbacks.append(LoggerCallback(frequency=logging_frequency))
callbacks.append(SummaryWriterCallback(frequency=summary_frequency,
                                       working_dir=working_dir))
callbacks.append(CheckpointWriterCallback(frequency=checkpoint_frequency,
                                          working_dir=working_dir,
                                          file_prefix=checkpoint_file_prefix))
callbacks.append(EvaluationCallback(
    frequency=evaluation_frequency,
    iterator=get_iterator(get_tf_mnist_data(data.train)),
    metrics=eval_metric, name='Train Accuracy'))
callbacks.append(EvaluationCallback(
    frequency=evaluation_frequency,
    iterator=get_iterator(get_tf_mnist_data(data.validation)),
    metrics=eval_metric, name='Eval Accuracy'))
callbacks.append(EvaluationCallback(
    frequency=evaluation_frequency,
    iterator=get_iterator(get_tf_mnist_data(data.test)),
    metrics=eval_metric, name='Test Accuracy'))

learner.train(loss, get_iterator(get_tf_mnist_data(data.train)),
              optimizer=optimizer, max_iter=max_iter,
              loss_chg_tol=loss_chg_tol,
              loss_chg_iter_below_tol=loss_chg_iter_below_tol,
              initialization_option=True, callbacks=callbacks,
              working_dir=working_dir,
              checkpoint_file_prefix=checkpoint_file_prefix,
              restore_sequentially=restore_sequentially,
              save_trained=save_trained)
tf_mnist_test_data = get_tf_mnist_data(data.test)
test_predictions = learner.predict(get_iterator(tf_mnist_test_data, False), -1)
test_truth = tf_mnist_test_data[1]
print(np.mean(test_predictions == test_truth))
