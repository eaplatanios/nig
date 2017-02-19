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

import tensorflow as tf

from .. import ops

__author__ = 'eaplatanios'

__all__ = ['mean', 'accuracy', 'confusion_matrix', 'area_under_curve']


def _remove_squeezable_dimensions(predictions, labels, weights=None, name=None):
    """Squeezes the last dimension of either `predictions` or `labels` if their
    rank differs by `1`. If `weights` is provided, then if its rank is `1` more
    than the rank of `predictions`, its last dimension is squeezed.

    Notes:
        This method will use the static shape of the tensors if it is available.
        Otherwise, it will add graph operations to compute the rank of the
        tensors, which could result in a performance hit.

    Args:
        predictions (tf.Tensor): Tensor containing the predictions.
        labels (tf.Tensor): Tensor containing the labels.
        weights (tf.Tensor, optional): Optional tensor containing weights for
            each prediction-label pair. Defaults to `None`.
        name (str, optional): Optional name to use when creating the TensorFlow
            name scope for the created ops. Defaults to `None`.

    Returns:
        (tf.Tensor, tf.Tensor): Tuple containing the predictions and the labels
            tensors, potentially with their last dimension squeezed, if
            `weights` is `None`, or, otherwise:

        (tf.Tensor, tf.Tensor, tf.Tensor): Tuple containing the predictions, the
            labels, and the weights tensors, potentially with their last
            dimension squeezed.
    """
    name_scope_values = [predictions, labels]
    if weights is not None:
        name_scope_values.append(weights)
    with tf.name_scope(name, 'remove_squeezable_dimensions', name_scope_values):
        predictions = tf.convert_to_tensor(predictions)
        labels = tf.convert_to_tensor(labels)
        if weights is not None:
            weights = tf.convert_to_tensor(weights)

        # Squeeze predictions and/or labels if necessary.
        predictions_shape = predictions.get_shape()
        predictions_rank = predictions_shape.ndims
        labels_shape = labels.get_shape()
        labels_rank = labels_shape.ndims
        if predictions_rank is not None and labels_rank is not None:
            # Use static shape if known.
            rank_diff = predictions_rank - labels_rank
            if rank_diff == 1:
                predictions = tf.squeeze(predictions, [-1])
            elif rank_diff == -1:
                labels = tf.squeeze(labels, [-1])
        else:
            # Otherwise use dynamic shape.
            rank_diff = tf.rank(predictions) - tf.rank(labels)
            if predictions_rank is None \
                    or predictions_shape.dims[-1].is_compatible_with(1):
                predictions = tf.cond(
                    tf.equal(rank_diff, 1),
                    lambda: tf.squeeze(predictions, [-1]), lambda: predictions)
            if labels_rank is None \
                    or labels_shape.dims[-1].is_compatible_with(1):
                labels = tf.cond(
                    tf.equal(rank_diff, -1),
                    lambda: tf.squeeze(labels, [-1]), lambda: labels)
        predictions_shape.assert_is_compatible_with(labels_shape)
        if weights is None:
            return predictions, labels

        # Squeeze weights if necessary.
        weights_shape = weights.get_shape()
        weights_rank = weights_shape.ndims
        if predictions_rank is not None and weights_rank is not None:
            # Use static shape if known.
            rank_diff = predictions_rank - weights_rank
            if rank_diff == -1:
                weights = tf.squeeze(weights, [-1])
        elif weights_shape is None \
                or weights_shape.dims[-1].is_compatible_with(1):
            # Otherwise use dynamic shape.
            rank_diff = tf.rank(predictions) - tf.rank(weights)
            weights = tf.cond(
                tf.equal(rank_diff, -1),
                lambda: tf.squeeze(weights, [-1]), lambda: weights)
        return predictions, labels, weights


def _safe_div(numerator, denominator, name=None):
    """Divides two values, returning `0` if the denominator is `<= 0`.

    Args:
        numerator (tf.Tensor): Division numerator.
        denominator (tf.Tensor): Division denominator.
        name (str, optional): Optional name for the returned op. Defaults to
            `None`.

    Returns:
        tf.Tensor: Tensor containing zeros where `denominator <= 0` and
            `numerator / denominator` elsewhere.
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.divide(numerator, denominator),
        tf.zeros_like(denominator),
        name=name)


def _safe_scalar_div(numerator, denominator, name=None):
    """Divides two scalars, returning `0` if the denominator is equal to `0`.

    Args:
        numerator (tf.Tensor): Division numerator.
        denominator (tf.Tensor): Division denominator.
        name (str, optional): Optional name for the returned op. Defaults to
            `None`.

    Returns:
        tf.Tensor: Scalar tensor equal to `0` if `denominator == 0`, else equal
            to `numerator / denominator`.
    """
    numerator.get_shape().with_rank_at_most(1)
    denominator.get_shape().with_rank_at_most(1)
    return tf.cond(
        tf.equal(tf.constant(0.0, dtype=tf.float64), denominator),
        lambda: tf.constant(0.0, dtype=tf.float64),
        lambda: tf.divide(numerator, denominator),
        name=name)


def _process_predictions(predictions, labels, thresholds):
    """Processes the provided predictions, labels, and thresholds. The
    processing consists of the following two steps:

    1. If `thresholds` is not `None`, then the predictions are set to `0` (or
        `False) if their values are less than that of the predictions, and to
        `1` (or `True`) otherwise. Note that `thresholds` is broadcasted to the
        shape of `predictions`.
    2. If the data type of `predictions` is different than that of `labels`,
       then a cast operation is performed for `predictions`.

    Args:
        predictions (tf.Tensor): Tensor containing the predictions.
        labels (tf.Tensor): Tensor containing the labels.
        thresholds (tf.Tensor, optional): Optional tensor containing the
            thresholds. Defaults to `None`.

    Returns:
        tf.Tensor: Tensor containing the processed predictions.
    """
    if thresholds is not None:
        if labels.dtype == tf.bool:
            predictions = tf.greater(predictions, thresholds)
        else:
            predictions = tf.nn.relu(
                tf.sign(tf.subtract(predictions, thresholds)))
    if predictions.dtype != labels.dtype:
        return tf.cast(predictions, labels.dtype)
    return predictions


def _process_requested_ops(requested_ops):
    """Processes the requested ops to make sure they are supported. These are
    ops that the statistics/metrics methods can return.

    The supported ops are the following:
        - `value`: Op holding a previously computed value of the statistic, if
          `update` is being used too, or simply an op that computes and return
          the value of the statistic for the provided data.
        - `update`: Update op that can updated the previously computed value of
          the statistic given new data (note that the value is not replaced, but
          rather updated in order to reflect the *addition* of new data). This
          op can be useful for computing the value of the statistic over
          streaming data. If this op is requested, then both the `value` and the
          `update` op need to be requested too, for the returned ops to be
          useful.
        - `reset`: Reset op that can reset the value of the previously computed
          value of the statistic in order to start accumulating the statistic
          value over a potentially new set of data. If this op is requested,
          then both the `value` and the `update` op need to be requested too,
          for the returned ops to be useful.

    Args:
        requested_ops (list(str), tuple(str), set(str)): List, tuple, or set,
            containing the requested ops.

    Returns:
        list(str) or tuple(str) or set(str): List, tuple, or set, containing the
            processed requested ops.

    Raises:
        ValueError: If `requested_ops` contains any unsupported ops.
    """
    all_ops = ('value', 'update', 'reset')
    if requested_ops is None:
        return all_ops
    if not isinstance(requested_ops, tuple) \
            and not isinstance(requested_ops, list):
        requested_ops = [requested_ops]
    for op in requested_ops:
        if op not in all_ops:
            raise ValueError('Unsupported op "%s" requested.' % op)
    streaming = 'update' in requested_ops or 'reset' in requested_ops
    if streaming and ('value' not in requested_ops
                      or 'update' not in requested_ops
                      or 'reset' not in requested_ops):
        raise ValueError('If the "update" op or the "reset" op is requested, '
                         'then all of "value", "update", and "reset" ops must '
                         'be requested at the same time.')
    return requested_ops


def mean(values, weights=None, axis=None, values_collections=None,
         updates_collections=None, resets_collections=None, name='mean',
         requested_ops=None):
    """Computes the (weighted) mean of the given values over the specified
    dimension of the provided values tensor.

    This method creates two local variables, `total` and `count` that are used
    to compute the average of `values`. This average is ultimately returned as
    an idempotent operation that simply divides `total` by `count`.

    This method can return the following types of TensorFlow ops, depending on
    what was requested:
        - `value`: Op holding a previously computed value of the statistic, if
          `update` is being used too, or simply an op that computes and return
          the value of the statistic for the provided data.
        - `update`: Update op that can updated the previously computed value of
          the statistic given new data (note that the value is not replaced, but
          rather updated in order to reflect the *addition* of new data). This
          op can be useful for computing the value of the statistic over
          streaming data. If this op is requested, then both the `value` and the
          `update` op need to be requested too, for the returned ops to be
          useful.
        - `reset`: Reset op that can reset the value of the previously computed
          value of the statistic in order to start accumulating the statistic
          value over a potentially new set of data. If this op is requested,
          then both the `value` and the `update` op need to be requested too,
          for the returned ops to be useful.

    Args:
        values (tf.Tensor): Tensor containing the values whose mean is computed.
        weights (tf.Tensor, optional): Optional tensor with rank either 0, or
            the same as that of `values`, containing the weights to be used when
            computing the mean. This tensor must be broadcastable to the shape
            of `values` (i.e., all of its dimensions must be either `1`, or the
            same as the corresponding dimensions of `values`). If `weights` is
            `None`, weights default to `1`. Use weights of `0` to mask values.
            Defaults to `None`.
        axis (int, list(int), tuple(int), optional): Optional axis or axes along
            which to perform the mean reduction. If `None`, then the mean over
            the whole tensor is computed and a scalar is returned. Defaults to
            `None`.
        values_collections (list(str), tuple(str)): Optional list of collection
            names that the `value` op should be added to. Defaults to `None`.
        updates_collections (list(str), tuple(str)): Optional list of collection
            names that the `update` op should be added to. Defaults to `None`.
        resets_collections (list(str), tuple(str)): Optional list of collection
            names that the `reset` op should be added to. Defaults to `None`.
        name (str, optional): Optional name to use when creating the TensorFlow
            name scope for the created ops. Defaults to `'mean'`.
        requested_ops (str, list(str), tuple(str), set(str), optional): Optional
            string, list, tuple, or set, specifying the ops to return. Defaults
            to `None`, which corresponds to returning all supported ops. Note
            that if a string is used, then the resulting op is returned alone,
            and not as part of a dictionary (which is done for all other cases).

    Returns:
        tf.Operation: The requested op, if only one op was specified using a
            string value for the `requested_ops` argument, or, otherwise:

        dict(str, tf.Operation): Dictionary containing the requested ops.
            The keys in dictionary correspond to the `requested_ops` and the
            values to their corresponding TensorFlow ops.

    Raises:
        ValueError: If `weights` is not `None` and its shape is not
            broadcastable to that of `values`, if any of the provided
            collections is not a `list` or a `tuple`, or if `requested_ops`
            contains any unsupported ops.
    """
    multiple_requested_ops = \
        isinstance(requested_ops, tuple) or isinstance(requested_ops, list)
    requested_ops = _process_requested_ops(requested_ops)
    if isinstance(axis, int):
        axis = [axis]
    with tf.variable_scope(name, 'mean', (values, weights)):
        values = tf.to_float(values)
        if weights is None and axis is not None:
            weights_sum = tf.to_float(tf.reduce_prod(
                tf.gather(tf.shape(values), tf.convert_to_tensor(axis))))
        elif weights is None:
            weights_sum = tf.to_float(tf.size(values))
        else:
            weights = ops.broadcast_weights(tf.to_float(weights), values)
            values = tf.multiply(values, weights)
            weights_sum = tf.reduce_sum(weights, axis=axis)
        if axis is not None:
            local_vars_shape = values.get_shape().as_list()
            local_vars_shape = [dim for i, dim in enumerate(local_vars_shape)
                                if i not in axis]
        return_ops = dict()
        streaming = 'update' in requested_ops or 'reset' in requested_ops
        if streaming:
            if axis is not None:
                total = ops.create_local(shape=local_vars_shape, name='total')
                count = ops.create_local(shape=local_vars_shape, name='count')
            else:
                total = ops.create_local(shape=[], name='total')
                count = ops.create_local(shape=[], name='count')
            stored_value = _safe_div(total, count, name=name + '_safe_div')
            update_total_op = tf.assign_add(
                total, tf.reduce_sum(values, axis=axis))
            update_count_op = tf.assign_add(count, weights_sum)
            return_ops['value'] = stored_value
            return_ops['update'] = _safe_div(
                update_total_op, update_count_op, name='update')
            return_ops['reset'] = tf.group(
                tf.assign(total, tf.zeros_like(total)),
                tf.assign(count, tf.zeros_like(count)), name='reset')
            if values_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops['value'])
            if updates_collections is not None:
                tf.ops.add_to_collections(
                    names=updates_collections, value=return_ops['update'])
            if resets_collections is not None:
                tf.ops.add_to_collections(
                    names=resets_collections, value=return_ops['reset'])
        else:
            value = tf.reduce_sum(values, axis=axis)
            if weights is None:
                value = tf.cond(
                    tf.equal(0.0, weights_sum),
                    lambda: tf.zeros_like(value),
                    lambda: tf.divide(value, weights_sum),
                    name=name + '_safe_div')
            else:
                value = _safe_div(value, weights_sum, name=name + '_safe_div')
            return_ops['value'] = value
            if values_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops['value'])
        if multiple_requested_ops:
            return return_ops
        return return_ops[requested_ops[0]]


def accuracy(predictions, labels, thresholds=None, weights=None,
             macro_average=True, values_collections=None,
             updates_collections=None, resets_collections=None, name='accuracy',
             requested_ops=None):
    """Computes the (weighted) accuracy of a set of predictions given the true
    corresponding labels.

    This method can return the following types of TensorFlow ops, depending on
    what was requested:
        - `value`: Op holding a previously computed value of the statistic, if
          `update` is being used too, or simply an op that computes and return
          the value of the statistic for the provided data.
        - `update`: Update op that can updated the previously computed value of
          the statistic given new data (note that the value is not replaced, but
          rather updated in order to reflect the *addition* of new data). This
          op can be useful for computing the value of the statistic over
          streaming data. If this op is requested, then both the `value` and the
          `update` op need to be requested too, for the returned ops to be
          useful.
        - `reset`: Reset op that can reset the value of the previously computed
          value of the statistic in order to start accumulating the statistic
          value over a potentially new set of data. If this op is requested,
          then both the `value` and the `update` op need to be requested too,
          for the returned ops to be useful.

    Notes:
        If `thresholds` is a `tuple` or `list`, then the resulting `value` and
        `update` ops have one more dimension. Their first dimension corresponds
        to the index in that `tuple` or `list` (i.e., to the threshold used, if
        the result for multiple thresholds was requested).

    Args:
        predictions (tf.Tensor): Tensor containing the predictions.
        labels (tf.Tensor): Tensor containing the corresponding labels and whose
            shape must match that of `predictions`.
        thresholds (float, tf.Tensor, tuple(float), list(float),
                    tuple(tf.Tensor), list(tf.Tensor), optional): Optional
            argument specifying the thresholds to use if `predictions` need to
            be thresholded to produce values of either `0` or `1` and be
            comparable to `labels`. If it is a tuple or list, then the returned
            value and update ops first dimension corresponds to the index in
            list. Note that the tensors provided need to be broadcastable to the
            shape of `predictions`. Defaults to `None`.
        weights (tf.Tensor, optional): Optional tensor with rank either 0, or
            the same as that of `predictions`, containing the weights to be used
            when computing the accuracy. This tensor must be broadcastable to
            the shape of `predictions` (i.e., all of its dimensions must be
            either `1`, or the same as the corresponding dimensions of
            `predictions`). If `weights` is `None`, weights default to `1`. Use
            weights of `0` to mask values. Defaults to `None`.
        macro_average (bool, optional): Optional boolean value indicating
            whether to compute a macro average of the statistic. If `False`,
            then a micro average is computed, meaning that the counts over all
            possible labels are aggregated together and then divided to form the
            final accuracy. If `True`, a macro average is computed, meaning that
            the accuracy for each label (i.e., second dimension of
            `predictions`) is computed first, and those accuracies are then
            averaged to form the final accuracy.
        values_collections (list(str), tuple(str)): Optional list of collection
            names that the `value` op should be added to. Defaults to `None`.
        updates_collections (list(str), tuple(str)): Optional list of collection
            names that the `update` op should be added to. Defaults to `None`.
        resets_collections (list(str), tuple(str)): Optional list of collection
            names that the `reset` op should be added to. Defaults to `None`.
        name (str, optional): Optional name to use when creating the TensorFlow
            name scope for the created ops. Defaults to `'mean'`.
        requested_ops (str, list(str), tuple(str), set(str), optional): Optional
            string, list, tuple, or set, specifying the ops to return. Defaults
            to `None`, which corresponds to returning all supported ops. Note
            that if a string is used, then the resulting op is returned alone,
            and not as part of a dictionary (which is done for all other cases).

    Returns:
        tf.Operation: The requested op, if only one op was specified using a
            string value for the `requested_ops` argument, or, otherwise:

        dict(str, tf.Operation): Dictionary containing the requested ops.
            The keys in dictionary correspond to the `requested_ops` and the
            values to their corresponding TensorFlow ops.

    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, if
            `weights` is not `None` and its shape is not broadcastable to that
            of `predictions`, if any of the provided collections is not a `list`
            or a `tuple`, or if `requested_ops` contains any unsupported ops.
    """
    # TODO: What happens when the labels dimension is missing?
    # Process the thresholds argument.
    if isinstance(thresholds, tuple) or isinstance(thresholds, list):
        multiple_thresholds = True
    elif thresholds is not None:
        thresholds = [thresholds]
        multiple_thresholds = False
    else:
        multiple_thresholds = False

    with tf.variable_scope(name, 'accuracy', (predictions, labels, thresholds,
                                              weights)):
        # Reshape predictions, labels, thresholds, and weights.
        num_thresholds = 1 if thresholds is None else len(thresholds)
        predictions = tf.expand_dims(predictions, axis=0)
        predictions = tf.tile(predictions, multiples=[num_thresholds, 1, 1])
        labels = tf.expand_dims(labels, axis=0)
        labels = tf.tile(labels, multiples=[num_thresholds, 1, 1])
        if thresholds is not None:
            thresholds = [t if isinstance(t, tf.Tensor)
                          else tf.convert_to_tensor(t) for t in thresholds]
            thresholds = tf.stack(thresholds, axis=0)
            desired_rank = predictions.get_shape().ndims
            current_rank = thresholds.get_shape().ndims
            for _ in range(desired_rank - current_rank):
                thresholds = tf.expand_dims(thresholds, axis=-1)
        if weights is not None:
            weights = tf.expand_dims(weights, axis=0)
            weights = ops.broadcast_weights(tf.to_float(weights), predictions)

        def average_accuracy(accuracy):
            if not macro_average:
                return accuracy
            if weights is None:
                return tf.reduce_mean(accuracy, axis=1)
            label_weights = tf.reduce_sum(weights, axis=1)
            label_weights_sum = tf.reduce_sum(label_weights, axis=1)
            accuracy = tf.multiply(label_weights, weights)
            accuracy = tf.reduce_sum(accuracy, axis=1)
            return tf.divide(accuracy, label_weights_sum)

        # Threshold the predictions.
        predictions = _process_predictions(
            predictions=predictions, labels=labels, thresholds=thresholds)
        is_correct = tf.to_float(tf.equal(predictions, labels))
        axis = 1 if macro_average else [1, 2]
        return_ops = mean(
            values=is_correct, weights=weights, axis=axis,
            values_collections=None, updates_collections=None,
            resets_collections=None, name=name, requested_ops=requested_ops)
        multiple_requested_ops = \
            isinstance(requested_ops, tuple) or isinstance(requested_ops, list)
        if multiple_requested_ops:
            for name, op in return_ops.items():
                if name == 'value' or name == 'update':
                    op = average_accuracy(op)
                    if not multiple_thresholds:
                        op = op[0]
                    return_ops[name] = op
                    if name == 'value' and values_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=op)
                    if name == 'update' and updates_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=op)
                elif name == 'reset':
                    if resets_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=op)
                else:
                    raise ValueError('Unsupported op "%s" encountered.' % op)
        elif requested_ops == 'value' or requested_ops == 'update':
            return_ops = average_accuracy(return_ops)
            if requested_ops == 'value' and values_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
            if requested_ops == 'update' and updates_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
        elif requested_ops == 'reset':
            if resets_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
        else:
            raise ValueError('Unsupported op "%s" encountered.' % requested_ops)
        if not multiple_thresholds:
            return_ops = return_ops[0]
        return return_ops


def confusion_matrix(
        predictions, labels, thresholds=None, weights=None, axis=None,
        includes=None, values_collections=None, updates_collections=None,
        resets_collections=None, name='confusion_matrix', requested_ops=None):
    """Computes the true positive, false negatives, true negative, and false
    positive predictions, of a set of predictions given the true corresponding
    labels. Note that not all these statistics need to be computed, but rather
    any requested subset of them.

    This method can return the following types of TensorFlow ops, depending on
    what was requested:
        - `value`: Op holding a previously computed value of the statistic, if
          `update` is being used too, or simply an op that computes and return
          the value of the statistic for the provided data.
        - `update`: Update op that can updated the previously computed value of
          the statistic given new data (note that the value is not replaced, but
          rather updated in order to reflect the *addition* of new data). This
          op can be useful for computing the value of the statistic over
          streaming data. If this op is requested, then both the `value` and the
          `update` op need to be requested too, for the returned ops to be
          useful.
        - `reset`: Reset op that can reset the value of the previously computed
          value of the statistic in order to start accumulating the statistic
          value over a potentially new set of data. If this op is requested,
          then both the `value` and the `update` op need to be requested too,
          for the returned ops to be useful.

    Notes:
        If `thresholds` is a `tuple` or `list`, then the resulting `value` and
        `update` ops have one more dimension. Their first dimension corresponds
        to the index in that `tuple` or `list` (i.e., to the threshold used, if
        the result for multiple thresholds was requested).

    Args:
        predictions (tf.Tensor): Tensor containing the predictions.
        labels (tf.Tensor): Tensor containing the corresponding labels and whose
            shape must match that of `predictions`.
        thresholds (float, tf.Tensor, tuple(float), list(float),
                    tuple(tf.Tensor), list(tf.Tensor), optional): Optional
            argument specifying the thresholds to use if `predictions` need to
            be thresholded to produce values of either `0` or `1` and be
            comparable to `labels`. If it is a tuple or list, then the returned
            value and update ops first dimension corresponds to the index in
            list. Note that the tensors provided need to be broadcastable to the
            shape of `predictions`. Defaults to `None`.
        weights (tf.Tensor, optional): Optional tensor with rank either 0, or
            the same as that of `predictions`, containing the weights to be used
            when computing the accuracy. This tensor must be broadcastable to
            the shape of `predictions` (i.e., all of its dimensions must be
            either `1`, or the same as the corresponding dimensions of
            `predictions`). If `weights` is `None`, weights default to `1`. Use
            weights of `0` to mask values. Defaults to `None`.
        axis (int, list(int), tuple(int), optional): Optional axis or axes along
            which to perform the mean reduction. If `None`, then the mean over
            the whole tensor is computed and a scalar is returned. Defaults to
            `None`.
        includes (str, tuple(str), list(str), optional): Optional string, tuple
            of strings, or list of strings, specifying which statistics to
            include in the computation. The possible values for the string
            include:
                - `'tp'`: True positives.
                - `'fn'`: False negatives.
                - `'tn'`: True negatives.
                - `'fp'`: False positives.
            Defaults to `None`, meaning that all of the above statistics are
            included in the results.
        values_collections (list(str), tuple(str)): Optional list of collection
            names that the `value` op should be added to. Defaults to `None`.
        updates_collections (list(str), tuple(str)): Optional list of collection
            names that the `update` op should be added to. Defaults to `None`.
        resets_collections (list(str), tuple(str)): Optional list of collection
            names that the `reset` op should be added to. Defaults to `None`.
        name (str, optional): Optional name to use when creating the TensorFlow
            name scope for the created ops. Defaults to `'mean'`.
        requested_ops (str, list(str), tuple(str), set(str), optional): Optional
            string, list, tuple, or set, specifying the ops to return. Defaults
            to `None`, which corresponds to returning all supported ops. Note
            that if a string is used, then the resulting op is returned alone,
            and not as part of a dictionary (which is done for all other cases).

    Returns:
        tf.Operation: The requested op, if only one op was specified using a
            string value for the `requested_ops` argument, and only one
            statistic was specified using a string value for the `includes`
            argument, or:

        dict(str, tf.Operation): Dictionary containing the requested ops, if
            multiple ops were specified in the `requested_ops` argument, and
            only one statistic was specified using a string value for the
            `includes` argument. The keys in the dictionary correspond to the
            `requested_ops` and the values to their corresponding TensorFlow
            ops, for the one included statistic. Or:

        dict(str, tf.Operation): Dictionary containing the requested ops, if
            only one op was specified using a string value for the
            `requested_ops` argument, but multiple statistics were specified for
            the `includes` argument. The keys in the dictionary correspond to
            the statistic name in `includes` and the values to their
            corresponding TensorFlow ops, for the one requested op type. Or,
            otherwise:

        dict(str, dict(str, tf.Operation)): Dictionary containing the requested
            ops. The keys in the dictionary correspond to the statistic name in
            `includes` and the values to an nested dictionary with ops. The keys
            of the nested dictionary correspond to the `requested_ops` and the
            values to their corresponding TensorFlow ops.

    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, if
            `weights` is not `None` and its shape is not broadcastable to that
            of `predictions`, if any of the provided collections is not a `list`
            or a `tuple`, or if `requested_ops` contains any unsupported ops.
    """
    # TODO: Maybe this can be made faster by sorting the predictions?
    # Process the thresholds argument.
    if isinstance(thresholds, tuple) or isinstance(thresholds, list):
        multiple_thresholds = True
    elif thresholds is not None:
        thresholds = [thresholds]
        multiple_thresholds = False
    else:
        multiple_thresholds = False

    # Process the includes argument.
    all_includes = ['tp', 'fn', 'tn', 'fp']
    multiple_includes = True
    if includes is None:
        includes = all_includes
    elif isinstance(includes, str):
        multiple_includes = False
        if includes not in all_includes:
            raise ValueError('Invalid confusion matrix term: %s.' % includes)
        includes = [includes]
    else:
        for term in includes:
            if term not in all_includes:
                raise ValueError('Invalid confusion matrix term: %s.' % term)

    # Process the requested_ops argument.
    multiple_requested_ops = \
        isinstance(requested_ops, tuple) or isinstance(requested_ops, list)
    requested_ops = _process_requested_ops(requested_ops)

    # Process the axis argument.
    if isinstance(axis, int):
        axis = [axis]

    # Create and return the requested ops.
    with tf.variable_scope(name, 'confusion_matrix', (predictions, labels,
                                                      thresholds, weights)):
        if weights is None:
            predictions, labels = _remove_squeezable_dimensions(
                predictions=predictions, labels=labels)
        else:
            predictions, labels, weights = _remove_squeezable_dimensions(
                predictions=predictions, labels=labels, weights=weights)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        # Reshape predictions, labels, thresholds, and weights.
        num_thresholds = 1 if thresholds is None else len(thresholds)
        predictions = tf.expand_dims(predictions, axis=0)
        labels = tf.expand_dims(labels, axis=0)
        predictions = tf.tile(predictions, multiples=[num_thresholds, 1, 1])
        labels = tf.tile(labels, multiples=[num_thresholds, 1, 1])
        if thresholds is not None:
            thresholds = [t if isinstance(t, tf.Tensor)
                          else tf.convert_to_tensor(t) for t in thresholds]
            thresholds = tf.stack(thresholds, axis=0)
            desired_rank = predictions.get_shape().ndims
            current_rank = thresholds.get_shape().ndims
            for _ in range(desired_rank - current_rank):
                thresholds = tf.expand_dims(thresholds, axis=-1)
        if weights is not None:
            weights = tf.expand_dims(weights, axis=0)
            weights = ops.broadcast_weights(tf.to_float(weights), predictions)

        # Threshold the predictions.
        labels = tf.cast(labels, dtype=tf.bool)
        pos_predictions = _process_predictions(
            predictions=predictions, labels=labels, thresholds=thresholds)
        if 'fn' in includes or 'tn' in includes:
            neg_predictions = tf.logical_not(pos_predictions)
        pos_labels = labels
        if 'fp' in includes or 'tn' in includes:
            neg_labels = tf.logical_not(pos_labels)

        streaming = 'update' in requested_ops or 'reset' in requested_ops
        if streaming:
            # Determine the shape of the internal storage variables.
            if axis is not None:
                if multiple_thresholds:
                    axis = [i + 1 for i in axis]
                local_vars_shape = predictions.get_shape().as_list()
                local_vars_shape = [s for i, s in enumerate(local_vars_shape)
                                    if i not in axis]
            else:
                local_vars_shape = [num_thresholds]

        # Create the requested ops.
        return_ops = dict()
        for term in includes:
            return_ops[term] = dict()
            if term == 'tp':
                mask = tf.to_float(tf.logical_and(pos_predictions, pos_labels))
            elif term == 'fn':
                mask = tf.to_float(tf.logical_and(neg_predictions, pos_labels))
            elif term == 'tn':
                mask = tf.to_float(tf.logical_and(neg_predictions, neg_labels))
            elif term == 'fp':
                mask = tf.to_float(tf.logical_and(pos_predictions, neg_labels))
            else:
                raise ValueError('Invalid confusion matrix term: %s.' % term)
            if weights is not None:
                mask = tf.multiply(mask, weights)
            if not multiple_thresholds:
                mask = mask[0]
            if streaming:
                stored_value = ops.create_local(
                    shape=local_vars_shape, name=term)
                return_ops[term]['value'] = stored_value
                return_ops[term]['update'] = tf.assign_add(
                    stored_value, tf.reduce_sum(mask, axis=axis), name='update')
                return_ops[term]['reset'] = tf.assign(
                    stored_value, tf.zeros_like(stored_value), name='reset')
                if values_collections is not None:
                    tf.ops.add_to_collections(
                        names=values_collections, value=return_ops['value'])
                if updates_collections is not None:
                    tf.ops.add_to_collections(
                        names=updates_collections, value=return_ops['update'])
                if resets_collections is not None:
                    tf.ops.add_to_collections(
                        names=resets_collections, value=return_ops['reset'])
            else:
                value = tf.reduce_sum(mask, axis=axis, name=term)
                return_ops[term]['value'] = value
                if values_collections is not None:
                    tf.ops.add_to_collections(
                        names=values_collections, value=return_ops['value'])
            if not multiple_requested_ops:
                return_ops[term] = return_ops[term][requested_ops[0]]
        if multiple_includes:
            return return_ops
        return return_ops[includes[0]]


def area_under_curve(
        predictions, labels, weights=None, num_thresholds=200,
        macro_average=False, curve='pr', values_collections=None,
        updates_collections=None, resets_collections=None, name='auc',
        requested_ops=None):
    """Computes the approximate AUC via a Riemann sum.
    The `auc` function creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC. To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.
    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall values
    (computed using the aforementioned variables). The `num_thresholds` variable
    controls the degree of discretization with larger numbers of thresholds more
    closely approximating the true AUC. The quality of the approximation may vary
    dramatically depending on `num_thresholds`.
    For best results, `predictions` should be distributed approximately uniformly
    in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
    approximation may be poor if this is not the case.
    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `auc`.
    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
    Args:
      labels: A `bool` `Tensor` whose shape matches `predictions`.
      predictions: A floating point `Tensor` of arbitrary shape and whose values
        are in the range `[0, 1]`.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
      num_thresholds: The number of thresholds to use when discretizing the roc
        curve.
      metrics_collections: An optional list of collections that `auc` should be
        added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      curve: Specifies the name of the curve to be computed, 'ROC' [default] or
      'PR' for the Precision-Recall-curve.
      name: An optional variable_scope name.
    Returns:
      auc: A scalar `Tensor` representing the current area-under-curve.
      update_op: An operation that increments the `true_positives`,
        `true_negatives`, `false_positives` and `false_negatives` variables
        appropriately and whose value matches `auc`.
    Raises:
      ValueError: If `predictions` and `labels` have mismatched shapes, or if
        `weights` is not `None` and its shape doesn't match `predictions`, or if
        either `metrics_collections` or `updates_collections` are not a list or
        tuple.
    """
    curve = curve.lower()
    if curve != 'roc' and curve != 'pr':
        raise ValueError('Invalid curve type "%s". Supported types are "ROC" '
                         'and "PR".' % curve)

    with tf.variable_scope(name, 'auc', (predictions, labels, weights)):
        # Create the thresholds for the confusion matrix.
        epsilon = 1e-7  # Used to account for floating point imprecision.
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - epsilon] + thresholds + [1.0 + epsilon]

        # Reshape predictions, labels, thresholds, and weights.
        # num_thresholds = 1 if thresholds is None else len(thresholds)
        # predictions = tf.expand_dims(predictions, axis=0)
        # predictions = tf.tile(predictions, multiples=[num_thresholds, 1, 1])
        # labels = tf.expand_dims(labels, axis=0)
        # labels = tf.tile(labels, multiples=[num_thresholds, 1, 1])
        # thresholds = [tf.convert_to_tensor(t) for t in thresholds]
        # thresholds = tf.stack(thresholds, axis=0)
        # if weights is not None:
        #     weights = tf.expand_dims(weights, axis=0)
        #     weights = ops.broadcast_weights(tf.to_float(weights), predictions)

        axis = 1 if macro_average else [1, 2]
        confusion_matrix_ops = confusion_matrix(
            predictions, labels, thresholds=thresholds, weights=weights,
            axis=axis, includes=['tp', 'fn', 'tn', 'fp'],
            values_collections=None, updates_collections=None,
            resets_collections=None, name='confusion_matrix',
            requested_ops=requested_ops)

        def compute_auc(tp, fn, tn, fp):
            """Computes the ROC curve AUC or the PR curve AUC based on the
            confusion matrix counts."""
            recall = tf.divide(tp + epsilon, tp + fn + epsilon)
            if curve == 'roc':
                false_positive_rate = tf.divide(fp, fp + tn + epsilon)
                x = false_positive_rate
                y = recall
            elif curve == 'pr':
                precision = tf.divide(tp + epsilon, tp + fp + epsilon)
                x = recall
                y = precision
            else:
                raise ValueError('Invalid curve type "%s". Supported types are '
                                 '"ROC" and "PR".' % curve)
            return tf.reduce_sum(tf.multiply(
                x[:num_thresholds - 1] - x[1:],
                (y[:num_thresholds - 1] + y[1:]) / 2.0), axis=0,
                name='riemann_sum')

        def average_auc(auc):
            if not macro_average:
                return auc
            if weights is None:
                return tf.reduce_mean(auc, axis=0)
            label_weights = tf.reduce_sum(weights, axis=0)
            label_weights_sum = tf.reduce_sum(label_weights, axis=0)
            auc = tf.multiply(label_weights, weights)
            auc = tf.reduce_sum(auc, axis=0)
            return tf.divide(auc, label_weights_sum)

        multiple_requested_ops = \
            isinstance(requested_ops, tuple) or isinstance(requested_ops, list)
        if multiple_requested_ops:
            return_ops = dict()
            for op in requested_ops:
                if op == 'value' or op == 'update':
                    tp = confusion_matrix_ops['tp'][op]
                    fn = confusion_matrix_ops['fn'][op]
                    tn = confusion_matrix_ops['tn'][op]
                    fp = confusion_matrix_ops['fp'][op]
                    auc = compute_auc(tp=tp, fn=fn, tn=tn, fp=fp)
                    return_ops[op] = average_auc(auc)
                    if op == 'value' and values_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=return_ops[op])
                    if op == 'update' and updates_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=return_ops[op])
                elif op == 'reset':
                    tp = confusion_matrix_ops['tp'][op]
                    fn = confusion_matrix_ops['fn'][op]
                    tn = confusion_matrix_ops['tn'][op]
                    fp = confusion_matrix_ops['fp'][op]
                    return_ops[op] = tf.group(tp, fn, tn, fp, name='reset')
                    if resets_collections is not None:
                        tf.ops.add_to_collections(
                            names=values_collections, value=return_ops[op])
                else:
                    raise ValueError('Unsupported op "%s" encountered.' % op)
        elif requested_ops == 'value' or requested_ops == 'update':
            tp = confusion_matrix_ops['tp']
            fn = confusion_matrix_ops['fn']
            tn = confusion_matrix_ops['tn']
            fp = confusion_matrix_ops['fp']
            auc = compute_auc(tp=tp, fn=fn, tn=tn, fp=fp)
            return_ops = average_auc(auc)
            if requested_ops == 'value' and values_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
            if requested_ops == 'update' and updates_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
        elif requested_ops == 'reset':
            tp = confusion_matrix_ops['tp']
            fn = confusion_matrix_ops['fn']
            tn = confusion_matrix_ops['tn']
            fp = confusion_matrix_ops['fp']
            return_ops = tf.group(tp, fn, tn, fp, name='reset')
            if resets_collections is not None:
                tf.ops.add_to_collections(
                    names=values_collections, value=return_ops)
        else:
            raise ValueError('Unsupported op "%s" encountered.' % requested_ops)
        return return_ops
