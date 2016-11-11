# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import abc
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from six import with_metaclass

from .models import Model, CombinedModel
from ..data.iterators import get_iterator
from ..evaluation.constraints import MutualExclusionConstraint
from ..evaluation.integrators import LogicIntegrator
from ..math.statistics.cross_validation import KFold
from ..utilities.functions import memoize
from ..utilities.tensorflow import graph_context

__author__ = 'eaplatanios'

__all__ = ['Learner', 'SimpleLearner', 'ValidationSetLearner',
           'CrossValidationLearner', 'NIGLearner']

__LEARNER_NOT_TRAINED_ERROR__ = 'The current learner has not been trained.'

logger = logging.getLogger(__name__)


def _process_callbacks(callbacks):
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return [callback.copy() for callback in callbacks]
    return callbacks.copy()


class Learner(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, new_graph=True, session=None,
                 predict_postprocess=None):
        if new_graph:
            self.graph = tf.Graph()
            if isinstance(models, list):
                self.models = [model.copy_to_graph(graph=self.graph)
                               for model in models]
                self.trainable = all(model.trainable for model in self.models)
            else:
                self.models = models.copy_to_graph(graph=self.graph)
                self.trainable = self.models.trainable
        else:
            if isinstance(models, list):
                self.graph = models[0].graph
                if any(model.graph != self.graph for model in models):
                    raise ValueError('When `new_graph` is set to `False`, all '
                                     'models must lie on the same graph.')
                self.trainable = all(model.trainable for model in models)
            else:
                self.graph = models.graph
                self.trainable = models.trainable
            self.models = models
        self.initial_session = session
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
        if isinstance(self.models, list):
            self.train_iteration = [0] * len(self.models)
        else:
            self.train_iteration = 0

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

    def _init_session(self, option, saver, working_dir, ckpt_file_prefix,
                      feed_dict=None):
        if option is None:
            option = False
        if feed_dict is None:
            feed_dict = dict()
        if isinstance(option, bool):
            if option:
                if self.session is None:
                    self.session = tf.Session()
                self.session.run(
                    fetches=tf.initialize_variables(tf.trainable_variables()),
                    feed_dict=feed_dict)
                self.session.run(
                    fetches=tf.initialize_all_variables(), feed_dict=feed_dict)
            elif self.session is None:
                raise ValueError('When the initialization option is a boolean '
                                 'value set to `False`, a session needs to be '
                                 'provided.')
        elif saver is None:
            raise ValueError('When the initialization option is an integer, '
                             'indicating that a saved checkpoint should be '
                             'loaded, a saver must also be provided.')
        else:
            if self.session is None:
                self.session = tf.Session()
            if isinstance(option, int):
                Learner._load_checkpoint(
                    session=self.session, saver=saver, working_dir=working_dir,
                    file_prefix=ckpt_file_prefix, step=option)
            else:
                raise TypeError('Unsupported initialization type %s '
                                'encountered.' % type(option))

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

    def _postprocessed_output_ops(self):
        # TODO: Make able to deal with all supported outputs formats.
        outputs_ops = self.combined_model.outputs
        if not isinstance(outputs_ops, list):
            return self.predict_postprocess(outputs_ops)
        return list(map(lambda op: self.predict_postprocess(op), outputs_ops))

    @graph_context
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
        saver = tf.train.Saver(
            restore_sequentially=restore_sequentially,
            write_version=tf.train.SaverDef.V1)
        feed_dict = self.combined_model.get_feed_dict(data, is_train=False)
        self._init_session(
            option=ckpt, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
        return self.session.run(fetches=outputs_ops, feed_dict=feed_dict)

    @graph_context
    def predict_iterator(self, data, pipelines=None, yield_input_data=False,
                         ckpt=None, working_dir=os.getcwd(),
                         ckpt_file_prefix='ckpt', restore_sequentially=False):
        data = get_iterator(data=data, pipelines=pipelines, cycle=False)
        outputs_ops = self._postprocessed_output_ops()
        saver = tf.train.Saver(
            restore_sequentially=restore_sequentially,
            write_version=tf.train.SaverDef.V1)
        is_first_batch = True
        for data_batch in data:
            feed_dict = self.combined_model.get_feed_dict(
                data_batch, is_train=False)
            if is_first_batch:
                self._init_session(
                    option=ckpt, saver=saver, working_dir=working_dir,
                    ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
            is_first_batch = False
            if not yield_input_data:
                yield self.session.run(outputs_ops, feed_dict)
            else:
                yield data, self.session.run(outputs_ops, feed_dict)


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
            session=self.initial_session,
            predict_postprocess=self.predict_postprocess)

    @graph_context
    def train(self, data, pipelines=None, init_option=-1, callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False):
        if not self.trainable:
            raise ValueError('The model is not trainable.')
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
        batch_size = self.models.optimizer_opts.get('batch_size', None)
        max_iter = self.models.optimizer_opts.get('max_iter', 10000)
        abs_loss_chg_tol = self.models.optimizer_opts.get(
            'abs_loss_chg_tol', 1e-10)
        rel_loss_chg_tol = self.models.optimizer_opts.get(
            'rel_loss_chg_tol', 1e-3)
        loss_chg_iter_below_tol = self.models.optimizer_opts.get(
            'loss_chg_iter_below_tol', 5)
        data = get_iterator(
            data=data, batch_size=batch_size, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(
            restore_sequentially=restore_sequentially,
            write_version=tf.train.SaverDef.V1)
        model = self.models
        data_batch = data.next()
        feed_dict = model.get_feed_dict(data_batch, is_train=True)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
        for callback in callbacks:
            callback.initialize(self, model, None, working_dir, summary_writer)
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        step = 0
        while True:
            _, loss = self.session.run(
                fetches=[model.train_op, model.loss], feed_dict=feed_dict)
            for callback in callbacks:
                callback(self.session, feed_dict, loss, self.train_iteration)
            loss_diff = abs(prev_loss - loss)
            if loss_diff < abs_loss_chg_tol \
                    or abs(loss_diff / prev_loss) < rel_loss_chg_tol:
                iter_below_tol += 1
            else:
                iter_below_tol = 0
            if iter_below_tol >= loss_chg_iter_below_tol:
                logger.info('Loss value converged.')
                break
            if step >= max_iter - 1:
                logger.info('Maximum number of iterations reached.')
                break
            data_batch = data.next()
            feed_dict = model.get_feed_dict(data_batch, is_train=True)
            step += 1
            prev_loss = loss
            self.train_iteration += 1
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=self.train_iteration)

    def _train_external(self, data, pipelines=None, init_option=-1,
                        callbacks=None, working_dir=os.getcwd(),
                        ckpt_file_prefix='ckpt', restore_sequentially=False,
                        save_trained=False):
        data = get_iterator(
            data=data, batch_size=None, cycle=True, pipelines=pipelines)
        callbacks = _process_callbacks(callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(
            restore_sequentially=restore_sequentially,
            write_version=tf.train.SaverDef.V1)
        model = self.models
        feed_dict = model.get_feed_dict(data.next(), is_train=True)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
        for callback in callbacks:
            callback.initialize(self, model, None, working_dir, summary_writer)

        def _step_callback():
            """Returns a step callback function for the TensorFlow external
            optimizer interface that keeps track of the current step number
            internally."""
            def inner(variables):
                self.train_iteration += 1
            return inner
        step_callback = _step_callback()

        def _loss_callback():
            """Returns a loss callback function for the TensorFlow external
            optimizer interface that only gets evoked when the step callback
            defined above had updated its step. In order to do that, this
            function also keep an internal state, of the last step value of
            the step callback function, in which it was evoked."""
            def inner(*fetches):
                if inner.step != self.train_iteration:
                    inner.step = self.train_iteration
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
                file_prefix=ckpt_file_prefix, step=self.train_iteration)

    def loss(self, loss_op, data, pipelines=None):
        data = get_iterator(data=data, cycle=False, pipelines=pipelines)
        loss = 0.0
        for data_batch in data:
            feed_dict = self.models.get_feed_dict(data_batch, is_train=True)
            loss += self.session.run(loss_op, feed_dict)
        return loss / len(data)

    @property
    def combined_model(self):
        return self.models


class ValidationSetLearner(Learner):
    """Used for training multiple models that have the same input and predict
    the same quantities, using a validation data set to pick the best model."""
    def __init__(self, models, val_loss=None, session=None,
                 predict_postprocess=None):
        super(ValidationSetLearner, self).__init__(
            models=models if isinstance(models, list) else [models],
            new_graph=False, session=session,
            predict_postprocess=predict_postprocess)
        self._val_loss = val_loss
        self.best_model_index = 0
        self.best_learner = None

    def copy(self):
        return ValidationSetLearner(
            models=self.models, val_loss=self._val_loss,
            session=self.initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index, add_val_loss_op=False):
        model = self.models[model_index]
        learner = SimpleLearner(
            model=model, new_graph=True,
            predict_postprocess=self.predict_postprocess)
        if not add_val_loss_op:
            return learner
        if self._val_loss is None:
            val_loss_op = learner.models.loss
        else:
            with learner.graph.as_default(), tf.name_scope('val_loss'):
                val_loss_op = self._val_loss[model_index](
                    learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    def train(self, data, pipelines=None, val_data=None, init_option=-1,
              callbacks=None, working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False, parallel=True):
        if not self.trainable:
            raise ValueError('At least one of the models is not trainable.')
        if val_data is None:
            val_data = data
        val_data = get_iterator(data=val_data, batch_size=None, cycle=False)
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
                    restore_sequentially=restore_sequentially,
                    write_version=tf.train.SaverDef.V1)
            Learner._save_checkpoint(
                session=self.best_learner.session, saver=saver,
                working_dir=working_dir, file_prefix=ckpt_file_prefix,
                step=self.best_learner.train_iteration)

    @property
    def combined_model(self):
        if self.best_learner is None:
            raise ValueError(__LEARNER_NOT_TRAINED_ERROR__)
        return self.best_learner.models

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
    def __init__(self, models, val_loss=None, session=None,
                 predict_postprocess=None):
        super(CrossValidationLearner, self).__init__(
            models=models if isinstance(models, list) else [models],
            new_graph=False, session=session,
            predict_postprocess=predict_postprocess)
        self._val_loss = val_loss
        self.best_model_index = 0
        self.best_learner = None
        self._data_type = -1

    def copy(self):
        return CrossValidationLearner(
            models=self.models, val_loss=self._val_loss,
            session=self.initial_session,
            predict_postprocess=self.predict_postprocess)

    def _get_model_learner(self, model_index, add_val_loss_op=False):
        model = self.models[model_index]
        learner = SimpleLearner(
            model=model, new_graph=True,
            predict_postprocess=self.predict_postprocess)
        if not add_val_loss_op:
            return learner
        if self._val_loss is None:
            val_loss_op = learner.models.loss
        else:
            with learner.graph.as_default(), tf.name_scope('val_loss'):
                val_loss_op = self._val_loss[model_index](
                    learner.models.outputs, learner.models.train_outputs)
        return learner, val_loss_op

    @staticmethod
    def _get_fold_data(data, indices):
        """Used by the cross-validation learner."""
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return data[indices]
        if isinstance(data, tuple):
            return tuple(d[indices] for d in data)
        if isinstance(data, dict):
            return {k: v[indices] for k, v in data.items()}
        raise TypeError('Unsupported data type %s encountered.' % type(data))

    def train(self, data, pipelines=None, cross_val=None, init_option=-1,
              callbacks=None, working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False, parallel=True):
        if not self.trainable:
            raise ValueError('At least one of the models is not trainable.')
        if cross_val is None:
            cross_val = KFold(len(data[0]), k=10)
        if parallel:
            def _train_model(config):
                config[0].train(
                    data=self._get_fold_data(data, [config[3]]),
                    pipelines=pipelines, init_option=init_option,
                    callbacks=callbacks, working_dir=config[2],
                    ckpt_file_prefix=ckpt_file_prefix,
                    restore_sequentially=restore_sequentially,
                    save_trained=save_trained)
                return config[0].loss(
                    loss_op=config[1],
                    data=self._get_fold_data(data, config[4]),
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
                        data=self._get_fold_data(data, train_indices),
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
                        data=self._get_fold_data(data, val_indices),
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


class NIGLearner(Learner):
    def __init__(self, models, consensus_loss_weight=1.0, new_graph=True,
                 session=None, predict_postprocess=None):
        if any(model.uses_external_optimizer for model in models):
            raise ValueError('Only internal optimizers are supported for this '
                             'learner.')
        super(NIGLearner, self).__init__(
            models=models, new_graph=new_graph, session=session,
            predict_postprocess=predict_postprocess)
        if self.trainable:
            self.consensus_loss_weight = consensus_loss_weight
            with self.graph.as_default(), tf.name_scope('nig_learner'):
                self.num_models = len(self.models)
                num_outputs = tf.shape(self.models[0].outputs)[1]
                initial_trust = tf.constant(
                    value=np.eye(self.num_models), name='initial_trust',
                    dtype=tf.float32)
                initial_trust = tf.reshape(
                    initial_trust, shape=[1, self.num_models, self.num_models])
                initial_trust = tf.tile(
                    initial_trust, multiples=[num_outputs, 1, 1])
                self.trust = tf.Variable(
                    initial_value=initial_trust, trainable=False,
                    validate_shape=False, name='trust', dtype=tf.float32)
                consensus_losses = self._consensus_losses()
                for m in range(len(self.models)):
                    self.models[m] = self.models[m].update_loss(
                        loss=self.models[m].loss + consensus_losses[m],
                        graph=self.graph)

    def _consensus_losses(self):
        num_models = len(self.models)
        # TODO: Need to deal with other model outputs formats.
        outputs = tf.pack([model.outputs for model in self.models], axis=-1)
        outputs = tf.expand_dims(outputs, dim=-1)
        outputs = tf.tile(outputs, multiples=[1, 1, 1, num_models])
        disagreement = outputs - tf.transpose(outputs, perm=[0, 1, 3, 2])
        disagreement = tf.square(disagreement)
        trust = tf.reshape(self.trust, [-1, num_models, num_models])
        loss = tf.einsum('ijkl,jlk->k', disagreement, trust)
        loss /= tf.reduce_prod(
            tf.cast(tf.shape(disagreement), dtype=tf.float32)[:-1])
        loss = tf.reshape(loss, [num_models])
        return tf.unpack(self.consensus_loss_weight * loss, axis=0)

    def copy(self, new_graph=True):
        return NIGLearner(
            models=self.models,
            consensus_loss_weight=self.consensus_loss_weight,
            new_graph=new_graph, session=self.initial_session,
            predict_postprocess=self.predict_postprocess)

    def update_trust(self, trust):
        return self.session.run(self.trust.assign(trust))

    def _get_feed_dict(self, data, is_train=False):
        feed_dict = dict()
        for model in self.models:
            feed_dict.update(model.get_feed_dict(data, is_train=is_train))
        return feed_dict

    @graph_context
    def train(self, data, pipelines=None, init_option=-1,
              per_model_callbacks=None, combined_model_callbacks=None,
              working_dir=os.getcwd(), ckpt_file_prefix='ckpt',
              restore_sequentially=False, save_trained=False,
              labeled_data=None, unlabeled_data=None, unlabeled_pipelines=None):
        if not self.trainable:
            raise ValueError('At least one of the models is not trainable.')
        models = self.models
        batch_size = [model.optimizer_opts.get('batch_size', None)
                      for model in models]
        if any(size != batch_size[0] for size in batch_size):
            raise ValueError('All batch sizes must match.')
        batch_size = batch_size[0]
        max_iter = [model.optimizer_opts.get('max_iter', 10000)
                    for model in models]
        abs_loss_chg_tol = [model.optimizer_opts.get('abs_loss_chg_tol', 1e-10)
                            for model in models]
        rel_loss_chg_tol = [model.optimizer_opts.get('rel_loss_chg_tol', 1e-3)
                            for model in models]
        loss_chg_iter_below_tol = [
            model.optimizer_opts.get('loss_chg_iter_below_tol', 5)
            for model in models]
        data = get_iterator(
            data=data, batch_size=batch_size, cycle=True, pipelines=pipelines)
        if labeled_data is None:
            labeled_data = get_iterator(
                data=data, cycle=True, pipelines=pipelines)
        else:
            labeled_data = get_iterator(
                data=labeled_data, cycle=True, pipelines=pipelines)
        if unlabeled_data is not None:
            if unlabeled_pipelines is None:
                unlabeled_pipelines = pipelines
            unlabeled_data = get_iterator(
                data=unlabeled_data, cycle=True, pipelines=unlabeled_pipelines)
        per_model_callbacks = [_process_callbacks(per_model_callbacks)
                               for _ in self.models]
        combined_model_callbacks = _process_callbacks(combined_model_callbacks)
        summary_writer = tf.train.SummaryWriter(working_dir, self.graph)
        saver = tf.train.Saver(
            restore_sequentially=restore_sequentially,
            write_version=tf.train.SaverDef.V1)
        data_batch = data.next()
        feed_dict = self._get_feed_dict(data_batch, is_train=True)
        self._init_session(
            option=init_option, saver=saver, working_dir=working_dir,
            ckpt_file_prefix=ckpt_file_prefix, feed_dict=feed_dict)
        for m, callbacks in enumerate(per_model_callbacks):
            for callback in callbacks:
                callback.initialize(
                    self, self.models[m], 'model_%d' % m, working_dir,
                    summary_writer)
        for callback in combined_model_callbacks:
            callback.initialize(
                self, self.combined_model, 'combined_model', working_dir,
                summary_writer)
        untrained_models = list(range(len(models)))
        prev_loss = [sys.float_info.max] * len(models)
        iter_below_tol = [0] * len(models)
        step = 0
        while True:
            # TODO: Add trust estimation and update here.
            if step >= 100 and (step - 100) % 100 == 0:
                self.update_trust(self._estimate_trust(
                    labeled_data=labeled_data, unlabeled_data=unlabeled_data))
            fetches = [[models[m].train_op, models[m].loss]
                       for m in untrained_models]
            run_outputs = self.session.run(fetches=fetches, feed_dict=feed_dict)
            losses = []
            for i, m in enumerate(untrained_models[:]):
                loss = run_outputs[i][1]
                for callback in per_model_callbacks[m]:
                    callback(
                        self.session, feed_dict, loss, self.train_iteration[m])
                loss_diff = abs(prev_loss[m] - loss)
                if loss_diff < abs_loss_chg_tol[m] \
                        or abs(loss_diff / prev_loss[m]) < rel_loss_chg_tol[m]:
                    iter_below_tol[m] += 1
                else:
                    iter_below_tol[m] = 0
                if iter_below_tol[m] >= loss_chg_iter_below_tol[m]:
                    logger.info('Model %d finished training: Loss value '
                                'converged.' % m)
                    untrained_models.remove(m)
                elif step >= max_iter[m] - 1:
                    logger.info('Model %d finished training: Maximum number of '
                                'iterations reached.' % m)
                    untrained_models.remove(m)
                prev_loss[m] = loss
                self.train_iteration[m] += 1
                losses.append(loss)
            for callback in combined_model_callbacks:
                callback(
                    self.session, feed_dict, np.mean(losses),
                    max(self.train_iteration))
            if len(untrained_models) == 0:
                logger.info('All models have finished training.')
                break
            data_batch = data.next()
            feed_dict = self._get_feed_dict(data_batch, is_train=True)
            step += 1
        if save_trained:
            Learner._save_checkpoint(
                session=self.session, saver=saver, working_dir=working_dir,
                file_prefix=ckpt_file_prefix, step=step)

    def _estimate_error_rates(self, labeled_data=None, unlabeled_data=None,
                              integrate_data=True, integrator=None,
                              jvm_options=None, return_predictions=False):
        if integrator is None:
            integrator = LogicIntegrator()
        if jvm_options is None:
            jvm_options = ['-Xmx12G']
        predicted_instances = []
        observed_instances = []
        model_predictions = []
        if labeled_data is not None:
            labeled_fetches = [[model.outputs, model.train_outputs]
                               for model in self.models]
            labeled_feed_dict = self._get_feed_dict(
                data=labeled_data.next(), is_train=True)
            outputs = self.session.run(
                fetches=labeled_fetches, feed_dict=labeled_feed_dict)
            for m, output in enumerate(outputs):
                if return_predictions:
                    model_predictions.append(output[0])
                for index, p in np.ndenumerate(output[0]):
                    predicted_instances.append(
                        (index[0], index[1], m, np.exp(p)))
                if m == 0:
                    if len(output[1].shape) > 1 and output[1].shape[1] > 1:
                        for index, l in np.ndenumerate(output[1]):
                            observed_instances.append((index[0], index[1], l))
                    else:
                        for index, l in np.ndenumerate(output[1]):
                            for label_id in range(output[0].shape[1]):
                                observed_instances.append(
                                    (index[0], label_id, l))
        index_offset = max(p[0] for p in predicted_instances) + 1
        if unlabeled_data is not None:
            unlabeled_fetches = [model.outputs for model in self.models]
            unlabeled_feed_dict = self._get_feed_dict(
                data=unlabeled_data.next(), is_train=False)
            outputs = self.session.run(
                fetches=unlabeled_fetches, feed_dict=unlabeled_feed_dict)
            for m, output in enumerate(outputs):
                if return_predictions:
                    model_predictions[m] = np.concatenate(
                        [model_predictions[m], output], axis=0)
                for index, p in np.ndenumerate(output):
                    predicted_instances.append(
                        (index[0] + index_offset, index[1], m, np.exp(p)))
        # TODO: Fix this hack.
        constraints = MutualExclusionConstraint([str(l) for l in range(10)])
        results = integrator.run(
            predicted=predicted_instances, observed=observed_instances,
            constraints=constraints, integrate_data=integrate_data,
            use_cli=False, jvm_options=jvm_options)
        predictions = np.stack(model_predictions, axis=0)
        if return_predictions and integrate_data:
            error_rates, integrated_data = results
            consensus = np.zeros(predictions.shape[1:])
            for instance in integrated_data:
                consensus[instance[0], int(instance[1])] = instance[2]
            results = error_rates, integrate_data, predictions, consensus
        elif return_predictions and not integrate_data:
            results = results, predictions
        return results

    def _estimate_trust(self, labeled_data=None, unlabeled_data=None,
                        integrator=None, jvm_options=None):
        error_rates, integrate_data, predictions, consensus = \
            self._estimate_error_rates(
                labeled_data=labeled_data, unlabeled_data=unlabeled_data,
                integrate_data=True, integrator=integrator,
                jvm_options=jvm_options, return_predictions=True)
        trust_shape = self.session.run(tf.shape(self.trust))
        trust = np.zeros(shape=trust_shape)
        method = 'consensus_biased'
        if method == 'uniform':
            for error_rate in error_rates:
                label = int(error_rate[0])
                trust[label, :, error_rate[1]] = error_rate[2]
            trust = -np.log(trust)
            trust -= np.min(trust, axis=2, keepdims=True)
            trust /= np.max(trust, axis=2, keepdims=True)
            # trust = 1 - trust
        elif method == 'consensus_biased':
            bias = 0.5
            consensus = np.tile(consensus[None], reps=[self.num_models, 1, 1])
            consensus = bias * predictions + (1 - bias) * consensus
            consensus = np.transpose(consensus, [2, 0, 1])[:, None, :, :]
            predictions = np.transpose(predictions, [2, 0, 1])[:, :, None, :]
            consensus = np.tile(consensus, [1, self.num_models, 1, 1])
            predictions = np.tile(predictions, [1, 1, self.num_models, 1])
            agreement = np.square(predictions - consensus)
            agreement = 1 - np.sum(agreement, axis=-1)
            trust = agreement
        logger.info(np.mean(trust, axis=1))
        return trust

    @property
    @memoize
    def combined_model(self):
        return CombinedModel(
            models=self.models,
            weights=tf.reduce_mean(self.trust, reduction_indices=[1]),
            graph=self.graph)
