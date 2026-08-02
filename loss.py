import tensorflow as tf


def quantile_loss(r, y_hat, nu):
    """
    3rd and 4th terms in Eq (4) of the original paper
    :param r: bias of hyperplane
    :param y_hat: data / output we're operating on
    :param nu: parameter between [0, 1] controls trade off between maximizing the distance of the hyperplane from
        the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
    :return: the loss function value
    """
    return (1 / nu) * tf.reduce_mean(tf.maximum(0.0, r - y_hat)) - r


def regularization_loss(w, V):
    """
    1st and 2nd terms in Eq (4) of the original paper: 0.5 * ||w||^2_2 + 0.5 * ||V||^2_F.

    Note these are *squared* norms. Using the plain (unsquared) norm makes the objective
    homogeneous of degree 1 in the scale of `w`, which - because the quantile term is
    negative at its optimum - leaves the objective unbounded below.

    :param w: weight matrix from hidden to output layer
    :param V: weight matrix from input to hidden layer
    :return: the regularization term
    """
    return 0.5 * tf.reduce_sum(tf.square(w)) + 0.5 * tf.reduce_sum(tf.square(V))
