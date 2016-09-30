import abc
import numpy as np
import os
import six
import sys
import tensorflow as tf
from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from six import with_metaclass

from nig.data.converters import TupleToDictConverter
from nig.data.iterators import DataIterator, NPArrayIterator
from nig.math.statistics.cross_validation import KFold
from nig.utilities.generic import logger, raise_error
from nig.utilities.iterators import ZipIterator

__author__ = 'eaplatanios'

__LEARNER_NOT_TRAINED_ERROR__ = 'The current learner has not been trained.'


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
            keep_last=True, pipelines=pipelines)
    if isinstance(data, tuple):
        if pipelines is None:
            return ZipIterator([_process_data_element(
                data=d, batch_size=batch_size, cycle=cycle) for d in data])
        if len(data) != len(pipelines):
            raise ValueError('data length should match that of pipelines.')
        return ZipIterator([_process_data_element(
            data=d, batch_size=batch_size, cycle=cycle, pipelines=p)
                            for d, p in zip(data, pipelines)])
    if isinstance(data, dict):
        if pipelines is None:
            return ZipIterator([_process_data_element(
                data=d, batch_size=batch_size, cycle=cycle)
                                for d in data.values()], list(data.keys()))
        if len(data) != len(pipelines):
            raise ValueError('data length should match that of pipelines.')
        if isinstance(pipelines, dict):
            pipelines = [pipelines[k] for k in data.keys()]
            return ZipIterator(
                [_process_data_element(
                    data=d, batch_size=batch_size, cycle=cycle, pipelines=p)
                 for d, p in zip(data.values(), pipelines)],
                list(data.keys()))
    if not isinstance(data, DataIterator):
        raise_error(ValueError, 'Unsupported data format encountered.')
    return data.reset_copy(
        batch_size=batch_size, cycle=cycle, pipelines=pipelines)


def _process_data_element(data, batch_size=None, cycle=False, pipelines=None):
    if isinstance(data, np.ndarray):
        batch_size = batch_size if batch_size is not None else len(data)
        return NPArrayIterator(
            data, batch_size=batch_size, shuffle=False, cycle=cycle,
            keep_last=True, pipelines=pipelines)
    if isinstance(data, tuple):
        batch_size = batch_size if batch_size is not None else len(data[0])
        return NPArrayIterator(
            data, batch_size=batch_size, shuffle=False, cycle=cycle,
            keep_last=True, pipelines=pipelines)
    if isinstance(data, dict):
        batch_size = batch_size if batch_size is not None \
            else len(six.next(six.itervalues(data)))
        if pipelines is not None:
            if isinstance(pipelines, list):
                pipelines = [p | TupleToDictConverter(data.keys())
                             for p in pipelines]
            else:
                pipelines = pipelines | TupleToDictConverter(data.keys())
        else:
            pipelines = TupleToDictConverter(data.keys())
        return NPArrayIterator(
            data=tuple(data.values()), batch_size=batch_size, shuffle=False,
            cycle=cycle, keep_last=True, pipelines=pipelines)
    if not isinstance(data, DataIterator):
        raise_error(ValueError, 'Unsupported data format encountered.')
    return data.reset_copy(
        batch_size=batch_size, cycle=cycle, pipelines=pipelines)


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
                    raise_error(
                        ValueError, 'When \'new_graph\' is set to \'False\', '
                                    'all models need to lie on the same graph.')
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
                    raise_error(ValueError, 'The input ops shapes must be '
                                            'equal for all models.')
                if not self._equal_shapes(
                        self._get_shapes(model.outputs), output_shapes):
                    raise_error(ValueError, 'The output ops shapes must be '
                                            'equal for all models.')
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
                raise_error(ValueError, 'When the initialization option is a '
                                        'boolean value set to False, then a '
                                        'session needs to be provided.')
            return
        if saver is None:
            raise_error(ValueError, 'When the initialization option is an '
                                    'integer, indicating that a saved '
                                    'checkpoint should be loaded, then a saver '
                                    'must also be provided.')
        if self.session is None:
            self.session = tf.Session()
        if isinstance(option, int):
            self._load_checkpoint(self.session, saver, working_dir,
                                  ckpt_file_prefix, option)
        else:
            raise_error(ValueError, 'Unsupported initialization.')
        return

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
        ckpt_file = tf.train.latest_checkpoint(
            checkpoint_dir=working_dir, latest_filename=file_prefix)
        if ckpt_file is not None:
            saver.restore(sess=session, save_path=ckpt_file)
        else:
            logger.warn('The requested checkpoint file does not exist. All the '
                        'variables are initialized to their default values.')
            session.run(tf.initialize_all_variables())

    @abc.abstractmethod
    def train(self, data, *args, **kwargs):
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
            self, input_data, pipelines=None, ckpt=None,
            working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
            restore_sequentially=False):
        if not isinstance(input_data, np.ndarray):
            iterator = self.predict_iterator(
                input_data=input_data, pipelines=pipelines, ckpt=ckpt,
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
            self._combined_model().get_feed_dict(input_data, is_train=False))

    @_graph_context
    def predict_iterator(
            self, input_data, pipelines=None, yield_input_data=False, ckpt=None,
            working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
            restore_sequentially=False):
        input_data = _process_data(input_data, pipelines=pipelines, cycle=False)
        outputs_ops = self._postprocessed_output_ops()
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=ckpt, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for data_batch in input_data:
            feed_dict = self._combined_model().get_feed_dict(
                data_batch, is_train=False)
            if not yield_input_data:
                yield self.session.run(outputs_ops, feed_dict)
            else:
                yield input_data, self.session.run(outputs_ops, feed_dict)


class SimpleLearner(Learner):
    """Used for training a single TensorFlow model."""
    def __init__(self, model, new_graph=True, session=None,
                 predict_postprocess=None):
        if isinstance(model, list):
            raise_error(ValueError, 'Cannot construct a simple learner with '
                                    'multiple models.')
        if model.uses_external_optimizer:
            raise_error(ValueError, 'SimpleLearner cannot be used with an '
                                    'external optimizer. Use '
                                    'SimpleLearnerExternalOptimizer instead.')
        super(SimpleLearner, self).__init__(
            models=model, new_graph=new_graph, session=session,
            predict_postprocess=predict_postprocess)

    def copy(self, new_graph=True):
        return SimpleLearner(
            model=self.models, new_graph=new_graph,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    @_graph_context
    def train(self, data, pipelines=None, batch_size=None, max_iter=100000,
              loss_chg_tol=1e-3, loss_chg_iter_below_tol=5, init_option=-1,
              callbacks=None, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False):
        """

        Args:
            data (Iterator or tuple(np.ndarray)):
            pipelines:
            batch_size:
            max_iter:
            loss_chg_tol:
            loss_chg_iter_below_tol:
            init_option:
            callbacks:
            working_dir:
            ckpt_file_prefix:
            restore_sequentially:
            save_trained:

        Returns:

        """
        if not self.models.trainable:
            raise_error(ValueError, 'A model is trainable only if both a loss '
                                    'and an optimizer are provided when '
                                    'constructing it.')
        data = _process_data(data, batch_size, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for callback in callbacks:
            callback.initialize(self, summary_writer)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        for step in range(max_iter):
            data_batch = data.next()
            feed_dict = self.models.get_feed_dict(data_batch, is_train=True)
            _, loss = self.session.run(
                fetches=[self.models.train_op, self.models.loss],
                feed_dict=feed_dict)
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

    def loss(self, loss_op, data, pipelines=None):
        data = _process_data(data, cycle=False, pipelines=pipelines)
        loss = 0.0
        for data_batch in data:
            feed_dict = self.models.get_feed_dict(data_batch, is_train=True)
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    def _combined_model(self):
        return self.models

    def _output_ops(self):
        return self.models.outputs


class SimpleLearnerExternalOptimizer(Learner):
    """Used for training a single TensorFlow model."""
    def __init__(self, model, new_graph=True, session=None,
                 predict_postprocess=None):
        if isinstance(model, list):
            raise_error(ValueError, 'Cannot construct a simple learner with '
                                    'multiple models.')
        if not model.uses_external_optimizer:
            raise_error(ValueError, 'SimpleLearnerExternalOptimizer cannot be '
                                    'used with an external optimizer. Use '
                                    'SimpleLearner instead.')
        super(SimpleLearnerExternalOptimizer, self).__init__(
            models=model, new_graph=new_graph, session=session,
            predict_postprocess=predict_postprocess)

    def copy(self, new_graph=True):
        return SimpleLearnerExternalOptimizer(
            model=self.models, new_graph=new_graph,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    @_graph_context
    def train(self, data, pipelines=None, init_option=-1, callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False):
        if not self.models.trainable:
            raise_error(ValueError, 'A model is trainable only if both a loss '
                                    'and an optimizer are provided when '
                                    'constructing it.')
        data = _process_data(
            data=data, batch_size=None, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(restore_sequentially=restore_sequentially)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix)
        for callback in callbacks:
            callback.initialize(self, summary_writer)
        feed_dict = self.models.get_feed_dict(data.next(), is_train=True)

        def _step_callback():
            """Returns a step callback function for the TensorFlow external
            optimizer interface that keeps track of the current step number
            internally."""
            def inner(variables):
                inner.step += 1
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

        self.models.optimizer.minimize(
            session=self.session, feed_dict=feed_dict,
            fetches=[self.models.loss], loss_callback=loss_callback,
            step_callback=step_callback)
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=step_callback.step)

    def loss(self, loss_op, data, pipelines=None):
        data = _process_data(data, cycle=False, pipelines=pipelines)
        loss = 0.0
        for data_batch in data:
            feed_dict = self.models.get_feed_dict(data_batch, is_train=True)
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    def _combined_model(self):
        return self.models

    def _output_ops(self):
        return self.models.outputs


class ValidationSetLearner(Learner):
    # TODO: Make this work with external optimizers.
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
        self.best_model = 0
        self.best_learner = None

    def copy(self):
        return ValidationSetLearner(
            models=self.models, val_loss=self._val_loss,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index):
        model = self.models[model_index]
        if model.uses_external_optimizer:
            learner = SimpleLearnerExternalOptimizer(
                model=model, new_graph=True,
                predict_postprocess=self.predict_postprocess)
        else:
            learner = SimpleLearner(
                model=model, new_graph=True,
                predict_postprocess=self.predict_postprocess)
        with learner.graph.as_default():
            val_loss_op = self._val_loss[model_index].tf_op(
                learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def train(self, data, pipelines=None, val_data=None, batch_size=None,
              max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
              init_option=-1, callbacks=None, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False, parallel=True):
        if val_data is None:
            val_data = data
        data = _process_data(data, batch_size, cycle=True, pipelines=pipelines)
        val_data = _process_data(val_data, batch_size, cycle=False)
        learners, val_loss_ops = tuple(zip(
            *[self._get_model_learner(model_index)
              for model_index in range(len(self.models))]))
        if parallel:
            def _train_model(config):
                config[0].train(
                    data=data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    working_dir=config[2], ckpt_file_prefix=ckpt_file_prefix,
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
                    data=data, max_iter=max_iter,
                    loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    working_dir=os.path.join(
                        working_dir, 'model_' + str(model_index)),
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                val_losses[model_index] = learners[model_index].loss(
                    loss_op=val_loss_ops[model_index], data=val_data,
                    pipelines=pipelines)
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
            raise_error(ValueError, __LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models

    def _output_ops(self):
        if self.best_learner is None:
            raise_error(ValueError, __LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models.outputs


class CrossValidationLearner(Learner):
    # TODO: Make this work with external optimizers.
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
        self.best_model = 0
        self.best_learner = None
        self._data_type = -1

    def copy(self):
        return CrossValidationLearner(
            models=self.models, val_loss=self._val_loss,
            session=self._initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index):
        model = self.models[model_index]
        if model.uses_external_optimizer:
            learner = SimpleLearnerExternalOptimizer(
                model=model, new_graph=True,
                predict_postprocess=self.predict_postprocess)
        else:
            learner = SimpleLearner(
                model=model, new_graph=True,
                predict_postprocess=self.predict_postprocess)
        with learner.graph.as_default():
            with tf.name_scope('learner'):
                val_loss_op = self._val_loss[model_index].tf_op(
                    learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def _get_fold_data(self, data, indices):
        if self._data_type == 0:
            return data[indices]
        elif self._data_type == 1:
            return tuple(d[indices] for d in data)
        elif self._data_type == 2:
            return {k: v[indices] for k, v in data.items()}
        raise_error(ValueError, 'Unsupported data type provided.')

    def train(self, data, pipelines=None, batch_size=None, cross_val=None,
              max_iter=100000, loss_chg_tol=1e-3, loss_chg_iter_below_tol=5,
              init_option=-1, callbacks=None, working_dir=os.getcwd(),
              ckpt_file_prefix='ckpt', restore_sequentially=False,
              save_trained=False, parallel=True):
        self._data_type = -1
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self._data_type = 0
        elif isinstance(data, tuple):
            self._data_type = 1
        elif isinstance(data, dict):
            self._data_type = 2
        else:
            raise_error(ValueError, 'Unsupported data type provided.')
        if cross_val is None:
            cross_val = KFold(len(data[0]), k=10)
        if parallel:
            def _train_model(config):
                config[0].train(
                    data=self._get_fold_data(data, [config[3]]),
                    pipelines=pipelines, batch_size=batch_size,
                    max_iter=max_iter, loss_chg_tol=loss_chg_tol,
                    loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                    init_option=init_option, callbacks=callbacks,
                    working_dir=config[2], ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0].loss(
                    loss_op=config[1],
                    data=self._get_fold_data(data, config[4]),
                    pipelines=pipelines)
            learners = []
            for model_index in range(len(self.models)):
                for fold in range(len(cross_val)):
                    learners.append(self._get_model_learner(model_index))
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
                    learner, val_loss_op = self._get_model_learner(model_index)
                    learner.train(
                        data=self._get_fold_data(data, train_indices),
                        pipelines=pipelines, # batch_size=batch_size,
                        # max_iter=max_iter, loss_chg_tol=loss_chg_tol,
                        # loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                        init_option=init_option, callbacks=callbacks,
                        working_dir=os.path.join(
                            working_dir,
                            'model_%d_fold_%d' % (model_index, num_folds - 1)),
                        ckpt_file_prefix=ckpt_file_prefix,
                        restore_sequentially=restore_sequentially,
                        save_trained=save_trained)
                    val_losses[model_index] += learner.loss(
                        loss_op=val_loss_op,
                        data=self._get_fold_data(data, val_indices),
                        pipelines=pipelines)
                val_losses[model_index] /= num_folds
        self.best_model = np.argmin(val_losses)
        self.best_learner, _ = self._get_model_learner(self.best_model)
        self.graph = self.best_learner.graph
        if save_trained:
            self.best_learner.train(
                data=data, batch_size=batch_size, max_iter=max_iter,
                loss_chg_tol=loss_chg_tol,
                loss_chg_iter_below_tol=loss_chg_iter_below_tol,
                init_option=init_option, callbacks=callbacks,
                working_dir=working_dir, ckpt_file_prefix=ckpt_file_prefix,
                restore_sequentially=restore_sequentially,
                save_trained=save_trained)

    def _combined_model(self):
        if self.best_learner is None:
            raise_error(ValueError, __LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models

    def _output_ops(self):
        if self.best_learner is None:
            raise_error(ValueError, __LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models.outputs


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
