import abc
import numpy as np
import os
import sys
import tensorflow as tf
from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from six import with_metaclass

from nig.data.iterators import NPArrayIterator
from nig.math.statistics.cross_validation import KFold
from nig.utilities.generic import logger

__author__ = 'eaplatanios'


def _graph_context(func):
    def func_wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return func(self, *args, **kwargs)
    return func_wrapper


def _process_data(data, batch_size=None, cycle=False):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else len(data)
        return NPArrayIterator(data, batch_size=batch_size, shuffle=False,
                               cycle=cycle, keep_last=True)
    if isinstance(data, tuple):
        batch_size = batch_size if batch_size is not None else len(data[0])
        return NPArrayIterator(data, batch_size=batch_size, shuffle=False,
                               cycle=cycle, keep_last=True)
    if not isinstance(data, NPArrayIterator):
        raise ValueError('Unsupported data format encountered.')
    return data.reset_copy(batch_size=batch_size, cycle=cycle)


def _process_callbacks(callbacks):
    if callbacks is None:
        return []
    return [callback.copy() for callback in callbacks]


class Learner(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, session=None, predict_postprocess=None):
        self.graph = tf.Graph()
        if isinstance(models, list):
            self.models = [model.copy_to_graph(self.graph) for model in models]
        else:
            self.models = models.copy_to_graph(self.graph)
        self.session = session
        if isinstance(models, list):
            input_shapes = self._get_shapes(models[0].inputs)
            output_shapes = self._get_shapes(models[0].outputs)
            for model in models:
                if not self._equal_shapes(
                        self._get_shapes(model.inputs), input_shapes):
                    error_msg = 'The input ops shapes must be equal for all ' \
                                'models.'
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not self._equal_shapes(
                        self._get_shapes(model.outputs), output_shapes):
                    error_msg = 'The output ops shapes must be equal for all ' \
                                'models.'
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        self.predict_postprocess = (lambda x: x) \
            if predict_postprocess is None \
            else predict_postprocess

    @staticmethod
    def _get_shapes(ops):
        if isinstance(ops, list):
            return [op.get_shape() for op in ops]
        return ops.get_shape()

    @staticmethod
    def _equal_shapes(shape_1, shape_2):
        if len(shape_1) != len(shape_2):
            return False
        for dim_1, dim_2 in zip(shape_1.dims, shape_2.dims):
            if dim_1 is None and dim_2 is None:
                continue
            if dim_1 != dim_2:
                return False
        return True

    @_graph_context
    def _init_session(self, option, saver, working_dir, ckpt_file_prefix):
        if option is None:
            option = False
        if isinstance(option, bool):
            if option:
                if self.session is None:
                    self.session = tf.Session()
                self.session.run(tf.initialize_all_variables())
                return
            elif self.session is None:
                raise ValueError('When the initialization option is a boolean '
                                 'value set to False, then a session needs to '
                                 'be provided.')
            return
        if saver is None:
            raise ValueError('When the initialization option is an integer, '
                             'indicating that a saved checkpoint should be '
                             'loaded, then a saver must also be provided.')
        if self.session is None:
            self.session = tf.Session()
        if isinstance(option, int):
            self._load_checkpoint(self.session, saver, working_dir,
                                  ckpt_file_prefix, option)
        else:
            raise ValueError('Unsupported initialization.')
        return

    @staticmethod
    def _save_checkpoint(session, saver, working_dir, file_prefix, step):
        saver.save(
            sess=session, save_path=os.path.join(working_dir, file_prefix),
            global_step=step, latest_filename=file_prefix)

    @staticmethod
    def _load_checkpoint(session, saver, working_dir, file_prefix, step):
        if step > -1:
            ckpt_file = os.path.join(working_dir, file_prefix) + str(step)
            if os.path.isfile(ckpt_file):
                saver.restore(sess=session, save_path=ckpt_file)
        ckpt_file = tf.train.latest_checkpoint(
            checkpoint_dir=working_dir, latest_filename=file_prefix)
        if ckpt_file is not None:
            saver.restore(sess=session, save_path=ckpt_file)
        else:
            logger.warn('The requested checkpoint file does not exist. All the '
                        'variables are initialized to their default values.')
            session.run(tf.initialize_all_variables())

    @abc.abstractmethod
    def train(
            self, train_data, batch_size=None, init_option=-1, callbacks=None,
            working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
            restore_sequentially=False, save_trained=False):
        pass

    @abc.abstractmethod
    def _combined_model(self):
        pass

    @abc.abstractmethod
    def _output_ops(self):
        pass

    def _postprocessed_output_ops(self):
        outputs_ops = self._output_ops()
        if not isinstance(outputs_ops, list):
            return self.predict_postprocess(outputs_ops)
        return list(map(lambda op: self.predict_postprocess(op), outputs_ops))

    @_graph_context
    def predict(
            self, input_data, ckpt=None, working_dir=os.getcwd(),
            ckpt_file_prefix='ckpt', restore_sequentially=False):
        if not isinstance(input_data, np.ndarray):
            iterator = self.predict_iterator(
                input_data=input_data, ckpt=ckpt, working_dir=working_dir,
                ckpt_file_prefix=ckpt_file_prefix,
                restore_sequentially=restore_sequentially)
            predictions = next(iterator)
            for batch in iterator:
                predictions = np.concatenate([predictions, batch], axis=0)
            return predictions
        outputs_ops = self._postprocessed_output_ops()
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=ckpt, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        return self.session.run(
            outputs_ops, self._combined_model().get_feed_dict(input_data))

    @_graph_context
    def predict_iterator(
            self, input_data, ckpt=None, working_dir=os.getcwd(),
            ckpt_file_prefix='ckpt', restore_sequentially=False):
        input_data = _process_data(input_data, cycle=False)
        outputs_ops = self._postprocessed_output_ops()
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=ckpt, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for data_batch in input_data:
            yield self.session.run(
                outputs_ops, self._combined_model().get_feed_dict(data_batch))


class SimpleLearner(Learner):
    """Used for training a single TensorFlow model."""
    def __init__(self, model, session=None, predict_postprocess=None):
        if isinstance(model, list):
            error_msg = 'Cannot construct a simple learner with multiple ' \
                        'models.'
            logger.error(error_msg)
            raise ValueError(error_msg)
        super(SimpleLearner, self).__init__(model, session, predict_postprocess)

    @_graph_context
    def train(self, train_data, batch_size=None, max_iter=100000,
              loss_chg_tol=1e-3, loss_chg_iter_below_tol=5, init_option=-1,
              callbacks=None, run_metadata_freq=1000,
              trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False):
        """

        Args:
            train_data (Iterator or tuple(np.ndarray)):
            batch_size:
            max_iter:
            loss_chg_tol:
            loss_chg_iter_below_tol:
            init_option:
            callbacks:
            run_metadata_freq:
            trace_level (tf.RunOptions): Supported values include
                `tf.RunOptions.{NO_TRACE, SOFTWARE_TRACE HARDWARE_TRACE,
                FULL_TRACE}`.
            working_dir:
            ckpt_file_prefix:
            restore_sequentially:
            save_trained:

        Returns:

        """
        if not self.models.trainable:
            raise ValueError('A model is trainable only if both a  loss and an '
                             'optimizer are provided when constructing it.')
        train_data = _process_data(train_data, batch_size, cycle=True)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for callback in callbacks:
            callback.initialize(
                self.graph, self.models, summary_writer, working_dir)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        for step in range(max_iter):
            data_batch = train_data.next()
            feed_dict = self.models.get_feed_dict(data_batch[0], data_batch[1])
            if run_metadata_freq > 0 \
                    and (step + 1) % run_metadata_freq == 0:
                run_options = tf.RunOptions(trace_level=trace_level)
                run_metadata = tf.RunMetadata()
                _, loss = self.session.run(
                    [self.models.train_op, self.models.loss_op], feed_dict,
                    options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(
                    run_metadata=run_metadata, tag='step' + str(step),
                    global_step=step)
            else:
                _, loss = self.session.run(
                    [self.models.train_op, self.models.loss_op], feed_dict)
            for callback in callbacks:
                callback(self.session, feed_dict, loss, step)
            if abs((prev_loss - loss) / prev_loss) < loss_chg_tol:
                iter_below_tol += 1
            else:
                iter_below_tol = 0
            if iter_below_tol >= loss_chg_iter_below_tol:
                logger.info('Loss value converged.')
                break
            prev_loss = loss
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=max_iter)

    def loss(self, loss_op, data):
        """

        Args:
            loss_op:
            data (Iterator or tuple(np.ndarray)):

        Returns:

        """
        data = _process_data(data, cycle=False)
        loss = 0.0
        for data_batch in data:
            feed_dict = self.models.get_feed_dict(data_batch[0], data_batch[1])
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    def _combined_model(self):
        return self.models

    def _output_ops(self):
        return self.models.outputs


class ValidationSetLearner(Learner):
    """Used for training multiple models that have the same input and predict
    the same quantities, using a validation data set to pick the best model."""
    def __init__(self, models, val_loss, session=None,
                 predict_postprocess=None):
        super(ValidationSetLearner, self).__init__(
            models=models if isinstance(models, list) else [models],
            session=session, predict_postprocess=predict_postprocess)
        if val_loss is None:
            val_loss = [model.loss for model in self.models]
        elif not isinstance(val_loss, list):
            val_loss = [val_loss] * len(self.models)
        self._val_loss = val_loss
        self.best_model = 0
        self.best_learner = None

    def _get_model_learner(self, model_index):
        learner = SimpleLearner(
            model=self.models[model_index],
            predict_postprocess=self.predict_postprocess)
        with learner.graph.as_default():
            val_loss_op = self._val_loss[model_index].tf_op(
                learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def train(self, train_data, val_data=None, batch_size=None,
              max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
              init_option=-1, callbacks=None, run_metadata_freq=1000,
              trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False, parallel=True):
        if val_data is None:
            val_data = train_data
        train_data = _process_data(train_data, batch_size, cycle=True)
        val_data = _process_data(val_data, batch_size, cycle=False)
        learners, val_loss_ops = tuple(zip(
            *[self._get_model_learner(model_index)
              for model_index in range(len(self.models))]))
        if parallel:
            def _train_model(config):
                config[0].train(
                    train_data=train_data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    run_metadata_freq=run_metadata_freq,
                    trace_level=trace_level, working_dir=config[2],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0].loss(
                    loss_op=config[1], data=val_data)
            with closing(ThreadPool()) as pool:
                val_losses = pool.map(
                    _train_model,
                    [(learners[model_index], val_loss_ops[model_index],
                      os.path.join(working_dir, 'model_' + str(model_index)))
                     for model_index in range(len(self.models))])
                pool.terminate()
        else:
            val_losses = [sys.float_info.max for _ in range(len(self.models))]
            for model_index in range(len(self.models)):
                learners[model_index].train(
                    train_data=train_data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    run_metadata_freq=run_metadata_freq,
                    trace_level=trace_level,
                    working_dir=os.path.join(
                        working_dir, 'model_' + str(model_index)),
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                val_losses[model_index] = learners[model_index].loss(
                    loss_op=val_loss_ops[model_index], data=val_data)
        self.best_model = np.argmin(val_losses)
        self.best_learner = learners[self.best_model]
        self.graph = self.best_learner.graph
        if save_trained:
            with self.best_learner.graph.as_default():
                saver = tf.train.Saver(
                    restore_sequentially=restore_sequentially)
            Learner._save_checkpoint(
                session=self.best_learner.session, saver=saver,
                working_dir=working_dir, file_prefix=ckpt_file_prefix,
                step=max_iter)

    def _combined_model(self):
        if self.best_learner is None:
            error_msg = 'The current learner has not been trained yet.'
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self.best_learner.models

    def _output_ops(self):
        if self.best_learner is None:
            error_msg = 'The current learner has not been trained yet.'
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self.best_learner.models.outputs


# class CrossValidationLearner(Learner):
#     """Used for training multiple symbols that have the same input and predict
#     the same quantities, using cross-validation to pick the best model."""
#     def __init__(self, models, loss, optimizers, val_loss=None,
#                  loss_summary=False, grads_processor=None, session=None,
#                  inputs_dtype=tf.float64, outputs_dtype=tf.float64,
#                  output_shape=None, predict_postprocess=None):
#         super(CrossValidationLearner, self).__init__(
#             models=models if isinstance(models, list) else [models],
#             session=session, inputs_dtype=inputs_dtype,
#             outputs_dtype=outputs_dtype, output_shape=output_shape,
#             predict_postprocess=predict_postprocess)
#         if not isinstance(optimizers, list):
#             optimizers = [optimizers] * len(self.models)
#         if val_loss is None:
#             val_loss = loss
#         self.loss = loss
#         self.val_loss = val_loss
#         self.optimizers = optimizers
#         self.loss_summary = loss_summary
#         self.grads_processor = grads_processor
#         self.best_symbol = 0
#
#     def _get_symbol_learner(self, symbol_index):
#         learner = SimpleLearner(
#             model=self.models[symbol_index], loss=self.loss,
#             optimizer=self.optimizers[symbol_index],
#             loss_summary=self.loss_summary,
#             grads_processor=self.grads_processor,
#             inputs_dtype=self.inputs_dtype, outputs_dtype=self.outputs_dtype,
#             output_shape=self.output_shape,
#             predict_postprocess=self.predict_postprocess)
#         with learner.graph.as_default():
#             val_loss_op = self.val_loss.tf_op(
#                 learner.predictions_op, learner.outputs_op)
#         return learner, val_loss_op
#
#     def train(self, train_data, cross_val=None, batch_size=None,
#               max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
#               init_option=-1, callbacks=None, run_metadata_freq=1000,
#               trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
#               ckpt_file_prefix='ckpt', restore_sequentially=False,
#               save_trained=False, parallel=True):
#         if not isinstance(train_data, tuple) or len(train_data) != 2 \
#                 or not isinstance(train_data[0], np.ndarray) \
#                 or not isinstance(train_data[1], np.ndarray) \
#                 or len(train_data[0]) != len(train_data[1]):
#             raise ValueError('The training data provided to the '
#                              'cross-validation learner needs to be a tuple of '
#                              'two numpy arrays with matching first dimensions. '
#                              'The first array should contain the inputs and '
#                              'the second, the corresponding labels.')
#         if cross_val is None:
#             cross_val = KFold(len(train_data[0]), k=10)
#         if parallel:
#             def _train_symbol(config):
#                 config[0].train(
#                     train_data=(train_data[0][config[3], :],
#                                 train_data[1][config[3], :]),
#                     batch_size=batch_size, max_iter=max_iter,
#                     loss_chg_tol=loss_chg_tol,
#                     loss_chg_iter_below_tol=loss_chg_iter_below_tol,
#                     init_option=init_option, callbacks=callbacks,
#                     run_metadata_freq=run_metadata_freq,
#                     trace_level=trace_level,
#                     working_dir=config[2],
#                     ckpt_file_prefix=ckpt_file_prefix,
#                     restore_sequentially=restore_sequentially,
#                     save_trained=save_trained)
#                 return config[0].loss(
#                     loss_op=config[1],
#                     data=(train_data[0][config[4], :],
#                           train_data[1][config[4], :]))
#             learners = []
#             for sym_index in range(len(self.models)):
#                 for fold in range(len(cross_val)):
#                     learners.append(self._get_symbol_learner(sym_index))
#             learners, val_loss_ops = tuple(zip(*learners))
#             with closing(ThreadPool()) as pool:
#                 configs = []
#                 learner_index = 0
#                 for sym_index in range(len(self.models)):
#                     for fold, indices in enumerate(cross_val.reset_copy()):
#                         configs.append((
#                             learners[learner_index],
#                             val_loss_ops[learner_index],
#                             os.path.join(working_dir, 'symbol_%d_fold_%d'
#                                          % (sym_index, fold)),
#                             indices[0], indices[1]))
#                         learner_index += 1
#                 val_losses = pool.map(_train_symbol, configs, chunksize=1)
#                 pool.terminate()
#                 val_losses = [np.mean(l) for l
#                               in np.array_split(val_losses, len(self.models))]
#         else:
#             val_losses = [sys.float_info.max] * len(self.models)
#             for sym_index in range(len(self.models)):
#                 cross_val.reset()
#                 val_losses[sym_index] = 0.0
#                 num_folds = 0
#                 for train_indices, val_indices in cross_val:
#                     num_folds += 1
#                     learner, val_loss_op = self._get_symbol_learner(sym_index)
#                     learner.train(
#                         train_data=(train_data[0][train_indices, :],
#                                     train_data[1][train_indices, :]),
#                         batch_size=batch_size, max_iter=max_iter,
#                         loss_chg_tol=loss_chg_tol,
#                         loss_chg_iter_below_tol=loss_chg_iter_below_tol,
#                         init_option=init_option, callbacks=callbacks,
#                         run_metadata_freq=run_metadata_freq,
#                         trace_level=trace_level,
#                         working_dir=os.path.join(
#                             working_dir,
#                             'symbol_%d_fold_%d' % (sym_index, num_folds - 1)),
#                         ckpt_file_prefix=ckpt_file_prefix,
#                         restore_sequentially=restore_sequentially,
#                         save_trained=save_trained)
#                     val_losses[sym_index] += learner.loss(
#                         loss_op=val_loss_op,
#                         data=(train_data[0][val_indices, :],
#                               train_data[1][val_indices, :]))
#                 val_losses[sym_index] /= num_folds
#         self.best_symbol = np.argmin(val_losses)
#         if save_trained:
#             learner, _ = self._get_symbol_learner(self.best_symbol)
#             learner.train(
#                 train_data=train_data, batch_size=batch_size, max_iter=max_iter,
#                 loss_chg_tol=loss_chg_tol,
#                 loss_chg_iter_below_tol=loss_chg_iter_below_tol,
#                 init_option=init_option, callbacks=callbacks,
#                 run_metadata_freq=run_metadata_freq, trace_level=trace_level,
#                 working_dir=working_dir, ckpt_file_prefix=ckpt_file_prefix,
#                 restore_sequentially=restore_sequentially,
#                 save_trained=save_trained)
#
#     @_graph_context
#     def _output_ops(self):
#         return self.models[self.best_symbol](self.inputs_op)


# class SimpleNIGLearner(Learner):
#     # TODO: Add support for multiple combination functions.
#     def __init__(self, symbols, loss=None, disagreement=None, optimizer=None,
#                  loss_summary=False, grads_processor=None, session=None,
#                  inputs_dtype=tf.float64, outputs_dtype=tf.float64,
#                  output_shape=None, predict_postprocess=None):
#         super(SimpleNIGLearner, self).__init__(
#             symbols=symbols if isinstance(symbols, list) else [symbols],
#             session=session, inputs_dtype=inputs_dtype,
#             outputs_dtype=outputs_dtype, output_shape=output_shape,
#             predict_postprocess=predict_postprocess)
#         self.trainable = loss is not None and disagreement is not None \
#             and optimizer is not None
#         with self.graph.as_default():
#             self.consensus_predictions_op = tf.placeholder(
#                 outputs_dtype, [None] + self.output_shape)
#
#
# class NIGLearner(Learner):
#     def __init__(self, symbols, loss, disagreement, optimizers,
#                  loss_summary=False, grads_processor=None, session=None,
#                  inputs_dtype=tf.float64, outputs_dtype=tf.float64,
#                  output_shape=None, predict_postprocess=None):
#         super(NIGLearner, self).__init__(
#             symbols=symbols if isinstance(symbols, list) else [symbols],
#             session=session, inputs_dtype=inputs_dtype,
#             outputs_dtype=outputs_dtype, output_shape=output_shape,
#             predict_postprocess=predict_postprocess)
#         if not isinstance(optimizers, list):
#             optimizers = [optimizers] * len(self.symbols)
#         self.loss = loss
#         self.optimizers = optimizers
#         self.loss_summary = loss_summary
#         self.grads_processor = grads_processor

