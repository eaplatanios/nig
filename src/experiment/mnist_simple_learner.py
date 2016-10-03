import tensorflow as tf

from nig.data.iterators import NPArrayIterator
from nig.data.loaders import mnist
from nig.data.processors import *
from nig.learning.callbacks import *
from nig.learning.metrics import *
from nig.learning.learners import *
from nig.learning.optimizers import gradient_descent
from nig.learning.processors import norm_summary, norm_clipping
from nig.learning.models import MultiLayerPerceptron

__author__ = 'eaplatanios'

use_one_hot_encoding = False
architectures = [[], [5]]
activation = tf.nn.relu
batch_size = 100
max_iter = 1000
loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 100
summary_frequency = 100
checkpoint_frequency = 1000
evaluation_frequency = 1000
working_dir = os.path.join(os.getcwd(), 'run')
checkpoint_file_prefix = 'ckpt'
restore_sequentially = False
save_trained = True
optimizer = gradient_descent(1e-1, decay_rate=0.99, learning_rate_summary=True)
optimizer_opts = {'batch_size': batch_size,
                  'max_iter': max_iter,
                  'loss_chg_tol': loss_chg_tol,
                  'loss_chg_iter_below_tol': loss_chg_iter_below_tol}
# optimizer = tf.contrib.opt.ScipyOptimizerInterface
# optimizer_opts = {'options': {'maxiter': 10000}}
gradients_processor = None #norm_clipping(clip_norm=0.1) \
                      #| norm_summary(name='gradients/norm')

train_data, val_data, test_data = mnist.load('data', float_images=True)

inputs_pipeline = ColumnsExtractor(list(range(784)))
labels_pipeline = ColumnsExtractor(784)
if use_one_hot_encoding:
    labels_pipeline = labels_pipeline | DataTypeEncoder(np.int8) | OneHotEncoder(10)


def get_iterator(mnist_data, include_labels=True):
    pipelines = [inputs_pipeline]
    if include_labels:
        pipelines.append(labels_pipeline)
    return NPArrayIterator(mnist_data, batch_size, shuffle=False, cycle=False,
                           cycle_shuffle=False, keep_last=True,
                           pipelines=pipelines)

loss = CrossEntropyOneHotEncodingMetric() if use_one_hot_encoding \
    else CrossEntropyIntegerEncodingMetric()
eval_metric = AccuracyOneHotEncodingMetric() if use_one_hot_encoding \
    else AccuracyIntegerEncodingMetric()

models = [MultiLayerPerceptron(
    784, 10, architecture, activation=activation,
    softmax_output=use_one_hot_encoding, use_log=use_one_hot_encoding,
    train_outputs_one_hot=use_one_hot_encoding, loss=loss, loss_summary=True,
    optimizer=optimizer, optimizer_opts=optimizer_opts,
    grads_processor=gradients_processor) for architecture in architectures]

callbacks = [
    LoggerCallback(frequency=logging_frequency),
    SummaryWriterCallback(frequency=summary_frequency),
    RunMetaDataSummaryWriter(
        frequency=1000, trace_level=tf.RunOptions.FULL_TRACE),
    VariableStatisticsSummaryWriterCallback(
        frequency=200, variables='trainable'),
    CheckpointWriterCallback(
        frequency=checkpoint_frequency, file_prefix=checkpoint_file_prefix),
    EvaluationCallback(
        frequency=evaluation_frequency, iterator=get_iterator(train_data),
        metrics=eval_metric, name='eval/train'),
    EvaluationCallback(
        frequency=evaluation_frequency, iterator=get_iterator(val_data),
        metrics=eval_metric, name='eval/val'),
    EvaluationCallback(
        frequency=evaluation_frequency, iterator=get_iterator(test_data),
        metrics=eval_metric, name='eval/test')]

# learner = SimpleLearner(
#     model=models[0], predict_postprocess=lambda l: tf.argmax(l, 1))
learner = ValidationSetLearner(
    models=models, val_loss=loss, predict_postprocess=lambda l: tf.argmax(l, 1))

learner.train(
    # train_data=get_iterator(train_data), val_data=get_iterator(val_data),
    data=(inputs_pipeline(train_data), labels_pipeline(train_data)),
    # data=train_data, pipelines=[inputs_pipeline, labels_pipeline],
    val_data=(inputs_pipeline(val_data), labels_pipeline(val_data)),
    # cross_val=KFold(len(train_data), 5),
    init_option=True,
    callbacks=callbacks, working_dir=working_dir,
    ckpt_file_prefix=checkpoint_file_prefix,
    restore_sequentially=restore_sequentially, save_trained=save_trained,
    parallel=True)
test_predictions = learner.predict(
    get_iterator(test_data, False), ckpt=-1, working_dir=working_dir,
    ckpt_file_prefix=checkpoint_file_prefix)
test_truth = test_data[:, -1]
print('Best model: %d' % learner.best_model_index)
print(np.mean(test_predictions == test_truth))

# Test loading the best performing trained model using a simple learner
simple_learner = SimpleLearner(
    models[learner.best_model_index], predict_postprocess=lambda l: tf.argmax(l, 1))
simple_learner_test_predictions = simple_learner.predict(
    get_iterator(test_data, False), ckpt=-1, working_dir=working_dir,
    ckpt_file_prefix=checkpoint_file_prefix)
print(np.mean(simple_learner_test_predictions == test_truth))
