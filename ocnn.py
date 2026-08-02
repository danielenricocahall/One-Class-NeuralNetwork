from typing import Callable

import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

from loss import quantile_loss, regularization_loss

tf.random.set_seed(0)


class OneClassNeuralNetwork:

    def __init__(self,
                 input_dim: int,
                 hidden_layer_size: int,
                 r: float = 1.0,
                 g: Callable[[tf.Tensor], tf.Tensor] = tf.nn.sigmoid):
        """

        :param input_dim: number of input features
        :param hidden_layer_size: number of neurons in the hidden layer
        :param r: bias of hyperplane
        :param g: activation function
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = tf.Variable(r, dtype=tf.float32, trainable=False, name="r")
        self.g = g

    @property
    def w(self) -> tf.Variable:
        """weight matrix from the hidden to the output layer"""
        return self.model.get_layer("hidden_output").kernel

    @property
    def V(self) -> tf.Variable:
        """weight matrix from the input to the hidden units"""
        return self.model.get_layer("input_hidden").kernel

    def custom_ocnn_loss(self, nu: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def custom_hinge(_: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
            # `w` and `V` are read as live layer kernels so that autodiff actually sees the
            # regularization term. `r` is held fixed here - it is solved for separately, in
            # closed form, once per epoch (Theorem 1 of the paper).
            return regularization_loss(self.w, self.V) + quantile_loss(self.r, y_hat, nu)

        return custom_hinge

    def build_model(self):
        h_size = self.hidden_size
        model = Sequential()
        # `use_bias=False` throughout: the paper scores a point as exactly <w, g(Vx)>, with no
        # bias terms. An output bias in particular is fatal - it is unregularized and shifts the
        # score and `r` together, so the network can drive the objective down by making its
        # output constant, collapsing to a degenerate solution that separates nothing.
        input_hidden = Dense(h_size, input_dim=self.input_dim,
                             kernel_initializer=RandomNormal(mean=0.0, stddev=0.1),
                             use_bias=False,
                             name="input_hidden")
        model.add(input_hidden)
        model.add(Activation(self.g))

        # Define Dense layer from hidden to output
        hidden_output = Dense(1, name="hidden_output",
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.1),
                              use_bias=False)
        model.add(hidden_output)
        model.add(Activation("linear"))

        self.model = model

        return model

    def update_r(self, X: np.array, nu: float) -> None:
        """
        solves for r given the current (w, V): per Theorem 1 of the paper, the optimal r is the
        nu-quantile of the scores {<w, g(Vx_n)>}. `interpolation='lower'` is used because the
        empirical minimizer of the pinball loss is the lower quantile.

        :param X: input training data
        :param nu: parameter between [0, 1], see `train_model`
        """
        y_hat = self.model(X, training=False)
        self.r.assign(tfp.stats.percentile(y_hat, q=100 * nu, interpolation='lower'))

    def train_model(self, X: np.array, epochs: int = 50, nu: float = 1e-2, init_lr: float = 1e-2, save: bool = True):
        """
        builds and trains the model on the supplied input data

        :param X: input training data
        :param epochs: number of epochs to train for (default 50)
        :param nu: parameter between [0, 1] controls trade off between maximizing the distance of the hyperplane from
        the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
        :param init_lr: initial learning rate (default 1e-2)
        :param save: flag indicating if the model should be  (default True)
        :return: trained model and callback history
        """

        def r_metric(*args):
            return tf.convert_to_tensor(self.r)

        def w_norm(*args):
            return 0.5 * tf.reduce_sum(tf.square(self.w))

        def V_norm(*args):
            return 0.5 * tf.reduce_sum(tf.square(self.V))

        def quantile_loss_metric(*args):
            return quantile_loss(self.r, args[1], nu)

        r_metric.__name__ = 'r'

        w_norm.__name__ = 'w_norm'

        V_norm.__name__ = 'V_norm'

        quantile_loss_metric.__name__ = 'quantile_loss'

        def on_epoch_begin(epoch, logs):
            # alternating minimization: solve for r against the current (w, V) before each pass
            # of gradient descent on (w, V)
            self.update_r(X, nu)

        model = self.build_model()

        model.compile(optimizer=Adam(learning_rate=init_lr),
                      loss=self.custom_ocnn_loss(nu),
                      metrics=[r_metric, quantile_loss_metric, w_norm, V_norm], run_eagerly=True)

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        history = model.fit(X, np.zeros((X.shape[0],)),
                            batch_size=X.shape[0],
                            shuffle=True,
                            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_begin=on_epoch_begin)],
                            epochs=epochs)

        if save:
            import os
            from datetime import datetime
            if not os.path.exists('models'):
                os.mkdir('models')
            model_dir = f"models/ocnn_{datetime.now().strftime('%Y-%m-%d-%H:%M:%s')}"
            os.mkdir(model_dir)
            model.save(f"{model_dir}/model.h5")
            np.savez(f"{model_dir}/params.npz", w=self.w.numpy(), V=self.V.numpy(), nu=nu, r=self.r.numpy())

        return model, history

    def load_model(self, model_dir: str) -> "OneClassNeuralNetwork":
        """
        loads a pretrained model
        :param model_dir: directory where model and model params (w, V, r, and nu) are saved
        :return: loaded model
        """
        params = np.load(f'{model_dir}/params.npz')
        nu = params['nu'].tolist()
        self.r.assign(params['r'].tolist())
        self.model = load_model(f'{model_dir}/model.h5',
                                custom_objects={'custom_hinge': self.custom_ocnn_loss(nu)})
        return self.model
