import tensorflow as tf

def softmax(x, axis=-1):
    # when x is a 2 dimensional tensor
    e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    s = tf.reduce_sum(e, axis=axis, keepdims=True)
    return e / s