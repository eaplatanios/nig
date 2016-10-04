from __future__ import absolute_import
from __future__ import division

import abc
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from six import with_metaclass

from nig.data.iterators import DataIterator, NPArrayIterator, ZipDataIterator
from nig.learning.models import Model
from nig.math.statistics.cross_validation import KFold

__author__ = 'eaplatanios'

__LEARNER_NOT_TRAINED_ERROR__ = 'The current learner has not been trained.'

logger = logging.getLogger(__name__)


def _graph_context(func):
    def func_wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return func(self, *args, **kwargs)
    return func_wrapper


def _process_data(data, batch_size=None, cycle=False, pipelines=None):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else len(data)
        return NPArrayIterator(
            data, batch_size=batch_size, shuffle=False, cycle=cycle,
            cycle_shuffle=False, keep_last=True, pipelines=pipelines)
    if isinstance(data, tuple):
        return ZipDataIterator(
            iterators=[_process_data_element(data=d, batch_size=batch_size)
                       for d in data],
            keys=None, batch_size=batch_size, cycle=cycle, pipelines=pipelines)
    if isinstance(data, dict):
        if isinstance(pipelines, dict):
            pipelines = [pipelines[k] for k in data.keys()]
        return ZipDataIterator(
            iterators=[_process_data_element(data=d, batch_size=batch_size)
                       for d in data.values()],
            keys=list(data.keys()), batch_size=batch_size, cycle=cycle,
            pipelines=pipelines)
    if not isinstance(data, DataIterator) \
            and not isinstance(data, ZipDataIterator):
        raise TypeError('Unsupported data type %s encountered.' % type(data))
    return data.reset_copy(
        batch_size=batch_size, cycle=cycle, pipelines=pipelines)


def _process_data_element(data, batch_size=None):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else len(data)
        return NPArrayIterator(data=data, batch_size=batch_size)
    if not isinstance(data, DataIterator):
        raise TypeError('Unsupported data type %s encountered.' % type(data))
    return data.reset_copy(batch_size=batch_size)


def _get_fold_data(data, indices):
    """Used by the cross-validation learner."""
    if isinstance(data, list) or isinstance(data, np.ndarray):
        return data[indices]
    if isinstance(data, tuple):
        return tuple(d[indices] for d in data)
    if isinstance(data, dict):
        return {k: v[indices] for k, v in data.items()}
    raise TypeError('Unsupported data type %s encountered.' % type(data))


def _process_callbacks(callbacks):
    if callbacks is None:
        return []
    return [callback.copy() for callback in callbacks]


class Learner(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, new_graph=True, session=None,
                 predict_postprocess=None):
        if new_graph:
            self.graph = tf.Graph()
            if isinstance(models, list):
                self.models = [model.copy_to_graph(self.graph)
                               for model in models]
            else:
                self.models = models.copy_to_graph(self.graph)
        else:
            if isinstance(models, list):
                self.graph = models[0].graph
                if any(model.graph != self.graph for model in models):
                    raise ValueError('When `new_graph` is set to `False`, all '
                                     'models must lie on the same graph.')
            else:
                self.graph = models.graph
            self.models = models
        self._initial_session = session
        self.session = session
        if isinstance(models, list):
            input_shapes = self._get_shapes(models[0].inputs)
            output_shapes = self._get_shapes(models[0].outputs)
            for model in models:
                if not self._equal_shapes(
                        self._get_shapes(model.inputs), input_shapes):
                    raise ValueError('The input ops shapes must be equal for '
                                     'all models.')
                if not self._equal_shapes(
                        self._get_shapes(model.outputs), output_shapes):
                    raise ValueError('The output ops shapes must be equal for '
                                     'all models.')
        self.predict_postprocess = (lambda x: x) \
            if predict_postprocess is None \
            else predict_postprocess

    @abc.abstractmethod
    def copy(self):
        pass

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

    @staticmethod
    def _save_checkpoint(session, saver, working_dir, file_prefix, step=None):
        saver.save(
            sess=session, save_path=os.path.join(working_dir, file_prefix),
            global_step=step, latest_filename=file_prefix)

    @staticmethod
    def _load_checkpoint(session, saver, working_dir, file_prefix, step):
        if step > -1:
            ckpt_file = os.path.join(working_dir, file_prefix) + str(step)
            if os.path.isfile(ckpt_file):
                saver.restore(sess=session, save_path=ckpt_file)
                return
        # TODO: Check if filename/working_dir contain illegal characters.
        ckpt_file = tf.train.latest_checkpoint(
            checkpoint_dir=working_dir, latest_filename=file_prefix)
        if ckpt_file is not None:
            saver.restore(sess=session, save_path=ckpt_file)
            return
        logger.warn('The requested checkpoint file does not exist. All the '
                    'variables are initialized to their default values.')
        session.run(tf.initialize_all_variables())

    @abc.abstractmethod
    def train(self, data, pipelines=None, init_option=-1, callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False):
        pass

    @property
    @abc.abstractmethod
    def combined_model(self):
        pass

    @abc.abstractmethod
    def _output_ops(self):
        pass

    def _postprocessed_output_ops(self):
        outputs_ops = self._output_ops()
        if not isinstance(outputs_ops, list):
            return self.predict_postprocess(outputs_ops)
        return list(map(lambda op: self.predict_postprocess(op), outputs_ops))

    @abc.abstractmethod
    def predict(self, data, pipelines=None, ckpt=None, working_dir=os.getcwd(),
                ckpt_file_prefix='ckpt', restore_sequentially=False):
        pass

    @abc.abstractmethod
    def predict_iterator(self, data, pipelines=None, yield_input_data=False,
                         ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        pass


class SimpleLearner(Learner):
    """Used for training a single TensorFlow model."""
    def __init__(self, model, new_graph=True, session=None,
                 predict_postprocess=None):
        if not isinstance(model, Model):
            raise TypeError('Unsupported model type %s encountered.'
                            % type(model))
        super(SimpleLearner, self).__init__(
            models=model, new_graph=new_graph, session=session,
            predict_postprocess=predict_postprocess)

    def copy(self, new_graph=True):
        return SimpleLearner(
            model=self.models, new_graph=new_graph,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

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
                                 'value set to `False`, a session needs to be '
                                 'provided.')
        if saver is None:
            raise ValueError('When the initialization option is an integer, '
                             'indicating that a saved checkpoint should be '
                             'loaded, a saver must also be provided.')
        if self.session is None:
            self.session = tf.Session()
        if isinstance(option, int):
            self._load_checkpoint(
                self.session, saver, working_dir, ckpt_file_prefix, option)
        else:
            raise TypeError('Unsupported initialization type %s encountered.'
                            % type(option))

    @_graph_context
    def train(self, data, pipelines=None, init_option=-1, callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False):
        if not self.models.trainable:
            raise ValueError('The provided model is not trainable.')
        if self.models.uses_external_optimizer:
            _train = self._train_external
        else:
            _train = self._train_internal
        _train(
            data=data, pipelines=pipelines, init_option=init_option,
            callbacks=callbacks, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix,
            restore_sequentially=restore_sequentially,
            save_trained=save_trained)

    def _train_internal(self, data, pipelines=None, init_option=-1,
                        callbacks=None, working_dir=os.getcwd(),
                        ckpt_file_prefix='ckpt', restore_sequentially=False,
                        save_trained=False):
        supported_opts = {'batch_size', 'max_iter', 'loss_chg_tol',
                          'loss_chg_iter_below_tol'}
        provided_opts = self.models.optimizer_opts.keys()
        unsupported_opts = provided_opts - supported_opts
        if len(unsupported_opts) > 0:
            logger.warn('Ignoring unsupported optimizer options %s. Supported '
                        'options are %s.' % (unsupported_opts, supported_opts))
        batch_size = self.models.optimizer_opts.get('batch_size', None)
        max_iter = self.models.optimizer_opts.get('max_iter', 10000)
        loss_chg_tol = self.models.optimizer_opts.get('loss_chg_tol', 1e-3)
        loss_chg_iter_below_tol = self.models.optimizer_opts.get(
            'loss_chg_iter_below_tol', 5)
        data = _process_data(
            data=data, batch_size=batch_size, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        model = self.models
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        self.global_step = self.session.run(model.global_step)
        for callback in callbacks:
            callback.initialize(self, working_dir, summary_writer)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        for step in range(max_iter):
            data_batch = data.next()
            feed_dict = model.get_feed_dict(data_batch, is_train=True)
            _, loss, global_step = self.session.run(
                fetches=[model.train_op, model.loss, model.global_step],
                feed_dict=feed_dict)
            for callback in callbacks:
                callback(self.session, feed_dict, loss, self.global_step)
            if abs((prev_loss - loss) / prev_loss) < loss_chg_tol:
                iter_below_tol += 1
            else:
                iter_below_tol = 0
            if iter_below_tol >= loss_chg_iter_below_tol:
                logger.info('Loss value converged.')
                break
            prev_loss = loss
            self.global_step = global_step
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=self.global_step)

    def _train_external(self, data, pipelines=None, init_option=-1,
                        callbacks=None, working_dir=os.getcwd(),
                        ckpt_file_prefix='ckpt', restore_sequentially=False,
                        save_trained=False):
        data = _process_data(
            data=data, batch_size=None, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        model = self.models
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for callback in callbacks:
            callback.initialize(self, working_dir, summary_writer)
        feed_dict = model.get_feed_dict(data.next(), is_train=True)
        _incr_global_step = tf.assign_add(model.global_step, 1)

        def _step_callback():
            """Returns a step callback function for the TensorFlow external
            optimizer interface that keeps track of the current step number
            internally."""
            def inner(variables):
                inner.step += 1
                self.global_step = self.session.run(fetches=_incr_global_step)
            inner.step = 0
            return inner
        step_callback = _step_callback()

        def _loss_callback():
            """Returns a loss callback function for the TensorFlow external
            optimizer interface that only gets evoked when the step callback
            defined above had updated its step. In order to do that, this
            function also keep an internal state, of the last step value of
            the step callback function, in which it was evoked."""
            def inner(*fetches):
                if inner.step != step_callback.step:
                    inner.step = step_callback.step
                    for call in callbacks:
                        args = fetches + (inner.step,)
                        call(self.session, feed_dict, *args)
            inner.step = -1
            return inner
        loss_callback = _loss_callback()

        model.optimizer.minimize(
            session=self.session, feed_dict=feed_dict,
            fetches=[model.loss], loss_callback=loss_callback,
            step_callback=step_callback)
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=self.global_step)

    def loss(self, loss_op, data, pipelines=None):
        data = _process_data(data=data, cycle=False, pipelines=pipelines)
        loss = 0.0
        for data_batch in data:
            feed_dict = self.models.get_feed_dict(data_batch, is_train=True)
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    @property
    def combined_model(self):
        return self.models

    def _output_ops(self):
        return self.models.outputs

    @_graph_context
    def predict(self, data, pipelines=None, ckpt=None, working_dir=os.getcwd(),
                ckpt_file_prefix='ckpt', restore_sequentially=False):
        if not isinstance(data, np.ndarray):
            iterator = self.predict_iterator(
                data=data, pipelines=pipelines, ckpt=ckpt,
                working_dir=working_dir, ckpt_file_prefix=ckpt_file_prefix,
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
            outputs_ops,
            self.combined_model.get_feed_dict(data, is_train=False))

    @_graph_context
    def predict_iterator(self, data, pipelines=None, yield_input_data=False,
                         ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        data = _process_data(data=data, pipelines=pipelines, cycle=False)
        outputs_ops = self._postprocessed_output_ops()
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=ckpt, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for data_batch in data:
            feed_dict = self.combined_model.get_feed_dict(
                data_batch, is_train=False)
            if not yield_input_data:
                yield self.session.run(outputs_ops, feed_dict)
            else:
                yield data, self.session.run(outputs_ops, feed_dict)


class ValidationSetLearner(Learner):
    """Used for training multiple models that have the same input and predict
    the same quantities, using a validation data set to pick the best model."""
    def __init__(self, models, val_loss, session=None,
                 predict_postprocess=None):
        super(ValidationSetLearner, self).__init__(
            models=models if isinstance(models, list) else [models],
            new_graph=False, session=session,
            predict_postprocess=predict_postprocess)
        if val_loss is None:
            val_loss = [model.loss for model in self.models]
        elif not isinstance(val_loss, list):
            val_loss = [val_loss] * len(self.models)
        self._val_loss = val_loss
        self.best_model_index = 0
        self.best_learner = None

    def copy(self):
        return ValidationSetLearner(
            models=self.models, val_loss=self._val_loss,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index, add_val_loss_op=False):
        model = self.models[model_index]
        learner = SimpleLearner(
            model=model, new_graph=True,
            predict_postprocess=self.predict_postprocess)
        if not add_val_loss_op:
            return learner
        with learner.graph.as_default():
            with tf.name_scope('val_loss'):
                val_loss_op = self._val_loss[model_index].tf_op(
                    learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def train(self, data, pipelines=None, val_data=None, init_option=-1,
              callbacks=None, working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False, parallel=True):
        if val_data is None:
            val_data = data
        val_data = _process_data(data=val_data, batch_size=None, cycle=False)
        learners, val_loss_ops = tuple(zip(
            *[self._get_model_learner(
                model_index=model_index, add_val_loss_op=True)
              for model_index in range(len(self.models))]))
        if parallel:
            def _train_model(config):
                config[0].train(
                    data=data, pipelines=pipelines, init_option=init_option,
                    callbacks=callbacks, working_dir=config[2],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0].loss(
                    loss_op=config[1], data=val_data, pipelines=pipelines)
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
                    data=data, pipelines=pipelines, init_option=init_option,
                    callbacks=callbacks, working_dir=os.path.join(
                        working_dir, 'model_' + str(model_index)),
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                val_losses[model_index] = learners[model_index].loss(
                    loss_op=val_loss_ops[model_index], data=val_data,
                    pipelines=pipelines)
        self.best_model_index = np.argmin(val_losses)
        self.best_learner = learners[self.best_model_index]
        if save_trained:
            with self.best_learner.graph.as_default():
                saver = tf.train.Saver(
                    restore_sequentially=restore_sequentially)
            Learner._save_checkpoint(
                session=self.best_learner.session, saver=saver,
                working_dir=working_dir, file_prefix=ckpt_file_prefix,
                step=self.best_learner.global_step)

    @property
    def combined_model(self):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models

    def _output_ops(self):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models.outputs

    def predict(self, data, pipelines=None, ckpt=None, working_dir=os.getcwd(),
                ckpt_file_prefix='ckpt', restore_sequentially=False):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.predict(
            data=data, pipelines=pipelines, ckpt=ckpt, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix,
            restore_sequentially=restore_sequentially)

    def predict_iterator(self, data, pipelines=None, yield_input_data=False,
                         ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.predict(
            data=data, pipelines=pipelines, yield_input_data=yield_input_data,
            ckpt=ckpt, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix,
            restore_sequentially=restore_sequentially)


class CrossValidationLearner(Learner):
    """Used for training multiple symbols that have the same input and predict
    the same quantities, using cross-validation to pick the best model."""
    def __init__(self, models, val_loss, session=None,
                 predict_postprocess=None):
        super(CrossValidationLearner, self).__init__(
            models=models if isinstance(models, list) else [models],
            new_graph=False, session=session,
            predict_postprocess=predict_postprocess)
        if val_loss is None:
            val_loss = [model.loss for model in self.models]
        elif not isinstance(val_loss, list):
            val_loss = [val_loss] * len(self.models)
        self._val_loss = val_loss
        self.best_model_index = 0
        self.best_learner = None
        self._data_type = -1

    def copy(self):
        return CrossValidationLearner(
            models=self.models, val_loss=self._val_loss,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index, add_val_loss_op=False):
        model = self.models[model_index]
        learner = SimpleLearner(
            model=model, new_graph=True,
            predict_postprocess=self.predict_postprocess)
        if not add_val_loss_op:
            return learner
        with learner.graph.as_default():
            with tf.name_scope('val_loss'):
                val_loss_op = self._val_loss[model_index].tf_op(
                    learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def train(self, data, pipelines=None, cross_val=None, init_option=-1,
              callbacks=None, working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False, parallel=True):
        if cross_val is None:
            cross_val = KFold(len(data[0]), k=10)
        if parallel:
            def _train_model(config):
                config[0].train(
                    data=_get_fold_data(data, [config[3]]),
                    pipelines=pipelines, init_option=init_option,
                    callbacks=callbacks, working_dir=config[2],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0].loss(
                    loss_op=config[1],
                    data=_get_fold_data(data, config[4]),
                    pipelines=pipelines)
            learners = []
            for model_index in range(len(self.models)):
                for fold in range(len(cross_val)):
                    learners.append(self._get_model_learner(
                        model_index=model_index, add_val_loss_op=True))
            learners, val_loss_ops = tuple(zip(*learners))
            with closing(ThreadPool()) as pool:
                configs = []
                learner_index = 0
                for model_index in range(len(self.models)):
                    for fold, indices in enumerate(cross_val.reset_copy()):
                        configs.append((
                            learners[learner_index],
                            val_loss_ops[learner_index],
                            os.path.join(
                                working_dir,
                                'model_%d_fold_%d' % (model_index, fold)),
                            indices[0], indices[1]))
                        learner_index += 1
                val_losses = pool.map(_train_model, configs, chunksize=1)
                pool.terminate()
                val_losses = [np.mean(l) for l
                              in np.array_split(val_losses, len(self.models))]
        else:
            val_losses = [sys.float_info.max for _ in range(len(self.models))]
            for model_index in range(len(self.models)):
                cross_val.reset()
                val_losses[model_index] = 0.0
                num_folds = 0
                for train_indices, val_indices in cross_val:
                    num_folds += 1
                    learner, val_loss_op = self._get_model_learner(
                        model_index=model_index, add_val_loss_op=True)
                    learner.train(
                        data=_get_fold_data(data, train_indices),
                        pipelines=pipelines, init_option=init_option,
                        callbacks=callbacks,
                        working_dir=os.path.join(
                            working_dir,
                            'model_%d_fold_%d' % (model_index, num_folds - 1)),
                        ckpt_file_prefix=ckpt_file_prefix,
                        restore_sequentially=restore_sequentially,
                        save_trained=save_trained)
                    val_losses[model_index] += learner.loss(
                        loss_op=val_loss_op,
                        data=_get_fold_data(data, val_indices),
                        pipelines=pipelines)
                val_losses[model_index] /= num_folds
        self.best_model_index = np.argmin(val_losses)
        self.best_learner = self._get_model_learner(
            model_index=self.best_model_index, add_val_loss_op=False)
        if save_trained:
            self.best_learner.train(
                data=data, pipelines=pipelines, init_option=init_option,
                callbacks=callbacks, working_dir=working_dir,
                ckpt_file_prefix=ckpt_file_prefix,
                restore_sequentially=restore_sequentially,
                save_trained=save_trained)

    @property
    def combined_model(self):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models

    def _output_ops(self):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models.outputs

    def predict(self, data, pipelines=None, ckpt=None, working_dir=os.getcwd(),
                ckpt_file_prefix='ckpt', restore_sequentially=False):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.predict(
            data=data, pipelines=pipelines, ckpt=ckpt, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix,
            restore_sequentially=restore_sequentially)

    def predict_iterator(self, data, pipelines=None, yield_input_data=False,
                         ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.predict(
            data=data, pipelines=pipelines, yield_input_data=yield_input_data,
            ckpt=ckpt, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix,
            restore_sequentially=restore_sequentially)


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
