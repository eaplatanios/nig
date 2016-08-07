import tensorflow as tf

from nig.functions import pipeline


@pipeline(min_num_args=1)
def norm_summary(tensors, name=None):
    if not isinstance(tensors, list):
        tensors = [tensors]
    if name is None:
        name = '_'.join([t.op.name for t in tensors]) + '_norm'
    tensors_norm = tf.reduce_sum([tf.nn.l2_loss(tensor) for tensor in tensors],
                                 name=name)
    tf.scalar_summary(tensors_norm.op.name, tensors_norm)
    return tensors


@pipeline(min_num_args=2)
def norm_clipping(tensors, clip_norm):
    clipped_tensors = []
    for t in tensors:
        clipped_tensors.append(tf.clip_by_norm(t, clip_norm))
    return clipped_tensors


@pipeline(min_num_args=2)
def average_norm_clipping(tensors, clip_norm):
    clipped_tensors = []
    for t in tensors:
        clipped_tensors.append(tf.clip_by_average_norm(t, clip_norm))
    return clipped_tensors


@pipeline(min_num_args=2)
def global_norm_clipping(tensors, clip_norm):
    tensors, _ = tf.clip_by_global_norm(tensors, clip_norm)
    return tensors
