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


def graph_context(func):
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


def _process_optimizer(optimizer):
    if callable(optimizer):
        optimizer = optimizer()
    if not isinstance(optimizer, tf.train.Optimizer):
        raise ValueError('Unsupported optimizer encountered.')
    return optimizer


def _process_callbacks(callbacks):
    if callbacks is None:
        return []
    return [callback.copy() for callback in callbacks]


def _train_op(loss, optimizer, loss_summary=False, grads_processor=None):
    global_step = tf.contrib.framework.get_or_create_global_step()
    if loss_summary:
        tf.scalar_summary(loss.op.name, loss)
    if grads_processor is not None:
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(ys=loss, xs=trainable_vars)
        grads = grads_processor(grads)
        train_op = optimizer.apply_gradients(
            grads_and_vars=zip(grads, trainable_vars), global_step=global_step)
    else:
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op


class Learner(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, symbols, graph=None, session=None,
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64,
                 output_shape=None, predict_postprocess=None):
        self.symbols = symbols
        self.graph = tf.Graph() if graph is None else graph
        self.session = session
        self.inputs_dtype = inputs_dtype
        self.outputs_dtype = outputs_dtype
        if isinstance(symbols, list):
            self.input_shape = self.symbols[0].input_shape
            self.output_shape = self.symbols[0].output_shape
            for symbol in symbols:
                if symbol.input_shape != self.input_shape:
                    error_message = 'All symbols input shapes must be equal.'
                    logger.error(error_message)
                    raise ValueError(error_message)
                if symbol.output_shape != self.output_shape:
                    error_message = 'All symbols output shapes must be equal.'
                    logger.error(error_message)
                    raise ValueError(error_message)
        else:
            self.input_shape = self.symbols.input_shape
            self.output_shape = self.symbols.output_shape
        if output_shape is not None and isinstance(output_shape, list):
            self.output_shape = output_shape
        elif output_shape is not None:
            self.output_shape = [output_shape]
        with self.graph.as_default():
            self.inputs_op = tf.placeholder(inputs_dtype,
                                            [None] + self.input_shape)
            self.outputs_op = tf.placeholder(outputs_dtype,
                                             [None] + self.output_shape)
        self.predict_postprocess = (lambda x: x) \
            if predict_postprocess is None \
            else predict_postprocess

    @graph_context
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
            logger.warn('The requested checkpoint file does not exist. '
                        'All the variables are initialized to their '
                        'default values.')
            session.run(tf.initialize_all_variables())

    @abc.abstractmethod
    def train(self, train_data, batch_size=None, init_option=-1,
              callbacks=None, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False):
        pass

    @abc.abstractmethod
    def _predict_op(self):
        pass

    @graph_context
    def predict(self, input_data, ckpt=None, working_dir=os.getcwd(),
                ckpt_file_prefix='ckpt', restore_sequentially=False):
        if not isinstance(input_data, np.ndarray):
            iterator = self.predict_iterator(
                input_data=input_data, ckpt=ckpt,
                working_dir=working_dir,
                ckpt_file_prefix=ckpt_file_prefix,
                restore_sequentially=restore_sequentially)
            predictions = next(iterator)
            for batch in iterator:
                predictions = np.concatenate([predictions, batch], axis=0)
            return predictions
        predict_op = self.predict_postprocess(self._predict_op())
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(option=ckpt, saver=saver, working_dir=working_dir,
                           ckpt_file_prefix=ckpt_file_prefix)
        return self.session.run(predict_op, {self.inputs_op: input_data})

    @graph_context
    def predict_iterator(self, input_data, ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        input_data = _process_data(input_data, cycle=False)
        predict_op = self.predict_postprocess(self._predict_op())
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(option=ckpt, saver=saver, working_dir=working_dir,
                           ckpt_file_prefix=ckpt_file_prefix)
        for data_batch in input_data:
            yield self.session.run(predict_op, {self.inputs_op: data_batch})


class SimpleLearner(Learner):
    """Used for training a single TensorFlow model."""

    def __init__(self, symbol, loss=None, optimizer=None, loss_summary=False,
                 grads_processor=None, graph=None, session=None,
                 inputs_dtype=tf.float64, outputs_dtype=tf.float64,
                 output_shape=None, predict_postprocess=None):
        super(SimpleLearner, self).__init__(
            symbols=symbol, graph=graph, session=session,
            inputs_dtype=inputs_dtype, outputs_dtype=outputs_dtype,
            output_shape=output_shape, predict_postprocess=predict_postprocess)
        self.trainable = loss is not None and optimizer is not None
        with self.graph.as_default():
            self.predictions_op = self.symbols(self.inputs_op)
            if self.trainable:
                optimizer = _process_optimizer(optimizer)
                self.loss_op = loss.tf_op(self.predictions_op, self.outputs_op)
                self.train_op = _train_op(self.loss_op, optimizer, loss_summary,
                                          grads_processor)

    @graph_context
    def train(self, train_data, batch_size=None, max_iter=100000,
              loss_chg_tol=1e-3, loss_chg_iter_below_tol=5, init_option=-1,
              callbacks=None, run_metadata_freq=1000,
              trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False):
        """

        Args:
            loss:
            train_data (Iterator or tuple(np.ndarray)):
            optimizer:
            batch_size:
            max_iter:
            loss_chg_tol:
            loss_chg_iter_below_tol:
            init_option:
            callbacks:
            loss_summary:
            grads_processor:
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
        if not self.trainable:
            raise ValueError('A simple learner is trainable only if both a '
                             'loss and an optimizer are provided when '
                             'constructing it.')
        train_data = _process_data(train_data, batch_size, cycle=True)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for callback in callbacks:
            callback.initialize(
                self.graph, self.inputs_op, self.outputs_op,
                self.predictions_op, self.loss_op, summary_writer, working_dir)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        for step in range(max_iter):
            data_batch = train_data.next()
            feed_dict = {self.inputs_op: data_batch[0],
                         self.outputs_op: data_batch[1]}
            if run_metadata_freq > 0 \
                    and (step + 1) % run_metadata_freq == 0:
                run_options = tf.RunOptions(trace_level=trace_level)
                run_metadata = tf.RunMetadata()
                _, loss = self.session.run(
                    [self.train_op, self.loss_op], feed_dict,
                    options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(
                    run_metadata=run_metadata, tag='step' + str(step),
                    global_step=step)
            else:
                _, loss = self.session.run(
                    [self.train_op, self.loss_op], feed_dict)
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

    def _loss(self, loss_op, data):
        """

        Args:
            loss_op:
            data (Iterator or tuple(np.ndarray)):

        Returns:

        """
        data = _process_data(data, cycle=False)
        loss = 0.0
        for data_batch in data:
            feed_dict = {self.inputs_op: data_batch[0],
                         self.outputs_op: data_batch[1]}
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    def _predict_op(self):
        return self.predictions_op


class ValidationSetLearner(Learner):
    """Used for training multiple symbols that have the same input and predict
    the same quantities, using a validation data set to pick the best model."""
    def __init__(self, symbols, loss, optimizers, val_loss=None,
                 loss_summary=False, grads_processor=None, graph=None,
                 session=None, inputs_dtype=tf.float64,
                 outputs_dtype=tf.float64, output_shape=None,
                 predict_postprocess=None):
        super(ValidationSetLearner, self).__init__(
            symbols=symbols if isinstance(symbols, list) else [symbols],
            graph=graph, session=session, inputs_dtype=inputs_dtype,
            outputs_dtype=outputs_dtype, output_shape=output_shape,
            predict_postprocess=predict_postprocess)
        if not isinstance(optimizers, list):
            optimizers = [optimizers] * len(self.symbols)
        if val_loss is None:
            val_loss = loss
        self.loss = loss
        self.val_loss = val_loss
        self.optimizers = optimizers
        self.loss_summary = loss_summary
        self.grads_processor = grads_processor
        self.best_symbol = 0

    def _get_symbol_learner(self, symbol_index):
        learner = SimpleLearner(
            symbol=self.symbols[symbol_index], loss=self.loss,
            optimizer=self.optimizers[symbol_index],
            loss_summary=self.loss_summary,
            grads_processor=self.grads_processor,
            inputs_dtype=self.inputs_dtype, outputs_dtype=self.outputs_dtype,
            output_shape=self.output_shape,
            predict_postprocess=self.predict_postprocess)
        with learner.graph.as_default():
            learner.val_loss_op = self.val_loss.tf_op(
                learner.predictions_op, learner.outputs_op)
        return learner

    def train(self, train_data, val_data=None,
              batch_size=None, max_iter=100000, loss_chg_tol=1e-3,
              loss_chg_iter_below_tol=5, init_option=-1, callbacks=None,
              run_metadata_freq=1000, trace_level=tf.RunOptions.FULL_TRACE,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False, parallel=True):
        if val_data is None:
            val_data = train_data
        train_data = _process_data(train_data, batch_size, cycle=True)
        val_data = _process_data(val_data, batch_size, cycle=False)
        learners = [self._get_symbol_learner(sym_index)
                    for sym_index in range(len(self.symbols))]
        if parallel:
            def _train_symbol(state):
                state[0].train(
                    train_data=train_data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    run_metadata_freq=run_metadata_freq,
                    trace_level=trace_level, working_dir=state[1],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return state[0]._loss(
                    loss_op=state[0].val_loss_op, data=val_data)
            with closing(ThreadPool()) as pool:
                val_losses = pool.map(
                    _train_symbol,
                    [(learners[sym_index],
                      os.path.join(working_dir, 'symbol_' + str(sym_index)))
                     for sym_index in range(len(self.symbols))])
                pool.terminate()
        else:
            val_losses = [sys.float_info.max] * len(self.symbols)
            for sym_index in range(len(self.symbols)):
                learners[sym_index].train(
                    train_data=train_data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    run_metadata_freq=run_metadata_freq,
                    trace_level=trace_level,
                    working_dir=os.path.join(working_dir,
                                             'symbol_' + str(sym_index)),
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                val_losses[sym_index] = learners[sym_index]._loss(
                    loss_op=learners[sym_index].val_loss_op, data=val_data)
        self.best_symbol = np.argmin(val_losses)
        if save_trained:
            best_learner = learners[self.best_symbol]
            with best_learner.graph.as_default():
                saver = tf.train.Saver(
                    restore_sequentially=restore_sequentially)
            Learner._save_checkpoint(
                session=best_learner.session, saver=saver,
                working_dir=working_dir, file_prefix=ckpt_file_prefix,
                step=max_iter)

    @graph_context
    def _predict_op(self):
        return self.symbols[self.best_symbol](self.inputs_op)


class CrossValidationLearner(Learner):
    """Used for training multiple symbols that have the same input and predict
    the same quantities, using cross-validation to pick the best model."""
    def __init__(self, symbols, loss, optimizers, val_loss=None,
                 loss_summary=False, grads_processor=None, graph=None,
                 session=None, inputs_dtype=tf.float64,
                 outputs_dtype=tf.float64, output_shape=None,
                 predict_postprocess=None):
        super(CrossValidationLearner, self).__init__(
            symbols=symbols if isinstance(symbols, list) else [symbols],
            graph=graph, session=session, inputs_dtype=inputs_dtype,
            outputs_dtype=outputs_dtype, output_shape=output_shape,
            predict_postprocess=predict_postprocess)
        if not isinstance(optimizers, list):
            optimizers = [optimizers] * len(self.symbols)
        if val_loss is None:
            val_loss = loss
        self.loss = loss
        self.val_loss = val_loss
        self.optimizers = optimizers
        self.loss_summary = loss_summary
        self.grads_processor = grads_processor
        self.best_symbol = 0

    def _get_symbol_learner(self, symbol_index):
        learner = SimpleLearner(
            symbol=self.symbols[symbol_index], loss=self.loss,
            optimizer=self.optimizers[symbol_index],
            loss_summary=self.loss_summary,
            grads_processor=self.grads_processor,
            inputs_dtype=self.inputs_dtype, outputs_dtype=self.outputs_dtype,
            output_shape=self.output_shape,
            predict_postprocess=self.predict_postprocess)
        with learner.graph.as_default():
            learner.val_loss_op = self.val_loss.tf_op(
                learner.predictions_op, learner.outputs_op)
        return learner

    def train(self, train_data, cross_val=None, batch_size=None,
              max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
              init_option=-1, callbacks=None, run_metadata_freq=1000,
              trace_level=tf.RunOptions.FULL_TRACE, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False, parallel=True):
        if not isinstance(train_data, tuple) or len(train_data) != 2 \
                or not isinstance(train_data[0], np.ndarray) \
                or not isinstance(train_data[1], np.ndarray) \
                or len(train_data[0]) != len(train_data[1]):
            raise ValueError('The training data provided to the '
                             'cross-validation learner needs to be a tuple of '
                             'two numpy arrays with matching first dimensions. '
                             'The first array should contain the inputs and '
                             'the second, the corresponding labels.')
        # TODO: Make the cross_validation parameter compulsory.
        if cross_val is None:
            cross_val = KFold(len(train_data[0]), k=10)
        if parallel:
            def _train_symbol(config):
                config[0].train(
                    train_data=(train_data[0][config[2], :],
                                train_data[1][config[2], :]),
                    batch_size=batch_size, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    run_metadata_freq=run_metadata_freq,
                    trace_level=trace_level,
                    working_dir=config[1],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0]._loss(
                    loss_op=config[0].val_loss_op,
                    data=(train_data[0][config[3], :],
                          train_data[1][config[3], :]))
            learners = []
            for sym_index in range(len(self.symbols)):
                for fold in range(len(cross_val)):
                    learners.append(self._get_symbol_learner(sym_index))
            with closing(ThreadPool()) as pool:
                configs = []
                learner_index = 0
                for sym_index in range(len(self.symbols)):
                    for fold, indices in enumerate(cross_val.reset_copy()):
                        configs.append((
                            learners[learner_index],
                            os.path.join(working_dir, 'symbol_%d_fold_%d'
                                         % (sym_index, fold)),
                            indices[0], indices[1]))
                        learner_index += 1
                val_losses = pool.map(_train_symbol, configs, chunksize=1)
                pool.terminate()
                val_losses = [np.mean(l) for l
                              in np.array_split(val_losses, len(self.symbols))]
        else:
            val_losses = [sys.float_info.max] * len(self.symbols)
            for sym_index in range(len(self.symbols)):
                cross_val.reset()
                val_losses[sym_index] = 0.0
                num_folds = 0
                for train_indices, val_indices in cross_val:
                    num_folds += 1
                    learner = self._get_symbol_learner(sym_index)
                    learner.train(
                        train_data=(train_data[0][train_indices, :],
                                    train_data[1][train_indices, :]),
                        batch_size=batch_size, max_iter=max_iter,
                        loss_chg_tol=loss_chg_tol,
                        loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                        init_option=init_option, callbacks=callbacks,
                        run_metadata_freq=run_metadata_freq,
                        trace_level=trace_level,
                        working_dir=os.path.join(working_dir,
                                                 'symbol_%d_fold_%d'
                                                 % (sym_index, num_folds - 1)),
                        ckpt_file_prefix=ckpt_file_prefix,
                        restore_sequentially=restore_sequentially,
                        save_trained=save_trained)
                    val_losses[sym_index] += learner._loss(
                        loss_op=learner.val_loss_op,
                        data=(train_data[0][val_indices, :],
                              train_data[1][val_indices, :]))
                val_losses[sym_index] /= num_folds
        self.best_symbol = np.argmin(val_losses)
        if save_trained:
            learner = self._get_symbol_learner(self.best_symbol)
            learner.train(
                train_data=train_data, batch_size=batch_size, max_iter=max_iter,
                loss_chg_tol=loss_chg_tol,
                loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                init_option=init_option, callbacks=callbacks,
                run_metadata_freq=run_metadata_freq, trace_level=trace_level,
                working_dir=working_dir, ckpt_file_prefix=ckpt_file_prefix,
                restore_sequentially=restore_sequentially,
                save_trained=save_trained)

    @graph_context
    def _predict_op(self):
        return self.symbols[self.best_symbol](self.inputs_op)


# class TensorFlowMultiModelNIGLearner(Learner):
#     """Used for training multiple TensorFlow models that have the same input
#     and
#     predict the same quantities, using the NIG agreement-driven approach to
#     train them jointly and allow them to make predictions jointly."""
#     def train(self, models, train_data, eval_data=None, test_data=None):
#         pass
#
#     def predict(self, input_data, checkpoint=-1, session=None):
#         pass
#
#     def predict_iterator(self, input_data):
#         pass
