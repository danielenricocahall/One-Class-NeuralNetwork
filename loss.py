import tensorflow as tf


def quantile_loss(r, y, nu):
    """
    3rd term in Eq (4) of the original paper
    :param r: bias of hyperplane
    :param y: data / output we're operating on
    :param nu: parameter between [0, 1] controls trade off between maximizing the distance of the hyperplane from
        the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
    :return: the loss function value
    """
    return (1 / nu) * tf.reduce_mean(tf.maximum(0.0, r - y), axis=-1)
