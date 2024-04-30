import tensorflow as tf


def softmax(x, axis=-1):
    # when x is a 2 dimensional tensor
    e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    s = tf.reduce_sum(e, axis=axis, keepdims=True)
    return e / s


X = tf.constant([[0.6, 0.9, 0.9], [0.8, 1.9, 2.5]])
softmax(X)
