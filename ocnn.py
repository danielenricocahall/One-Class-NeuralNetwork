import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

from loss import quantile_loss


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class OneClassNeuralNetwork:

    def __init__(self, input_dim, hidden_layer_size, r=1.0):
        """

        :param input_dim: number of input features
        :param hidden_layer_size: number of neurons in the hidden layer
        :param r: bias of hyperplane
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = r

    def custom_ocnn_loss(self, nu, w, V):
        def custom_hinge(_, y_pred):
            loss = 0.5 * tf.reduce_sum(w ** 2) + 0.5 * tf.reduce_sum(V ** 2) + quantile_loss(self.r, y_pred, nu)
            self.r = tfp.stats.percentile(tf.reduce_max(y_pred, axis=1), q=100 * nu)
            return loss

        return custom_hinge

    def build_model(self):
        h_size = self.hidden_size
        model = Sequential()
        input_hidden = Dense(h_size, input_dim=self.input_dim, kernel_initializer="glorot_normal",  name="input_hidden")
        model.add(input_hidden)
        model.add(Activation("linear"))

        # Define Dense layer from hidden to output
        hidden_ouput = Dense(1, name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation("sigmoid"))

        w = input_hidden.get_weights()[0]
        V = hidden_ouput.get_weights()[0]

        return [model, w, V]

    def train_model(self, X, epochs=50, nu=1e-2, init_lr=1e-2, save=True):
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
            return self.r

        r_metric.__name__ = 'r'

        def quantile_loss_metric(*args):
            return quantile_loss(self.r, args[1], nu)

        quantile_loss_metric.__name__ = 'quantile_loss'

        [model, w, V] = self.build_model()

        model.compile(optimizer=Adam(lr=init_lr, decay=init_lr / epochs),
                      loss=self.custom_ocnn_loss(nu, w, V), metrics=[r_metric, quantile_loss_metric], run_eagerly=True)

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        history = model.fit(X, np.zeros((X.shape[0],)),
                            steps_per_epoch=1,
                            shuffle=True,
                            epochs=epochs)

        if save:
            import os
            from datetime import datetime
            if not os.path.exists('models'):
                os.mkdir('models')
            model_dir = f"models/ocnn_{datetime.now().strftime('%Y-%m-%d-%H:%M:%s')}"
            os.mkdir(model_dir)
            w = model.layers[0].get_weights()[0]
            V = model.layers[2].get_weights()[0]
            model.save(f"{model_dir}/model.h5")
            np.savez(f"{model_dir}/params.npz", w=w, V=V, nu=nu)

        return model, history

    def load_model(self, model_dir):
        """
        loads a pretrained model
        :param model_dir: directory where model and model params (w, V, and nu) are saved
        :param nu: same as nu described in train_model
        :return: loaded model
        """
        params = np.load(f'{model_dir}/params.npz')
        w = params['w']
        V = params['V']
        nu = params['nu'].tolist()
        model = load_model(f'{model_dir}/model.h5',
                           custom_objects={'custom_hinge': self.custom_ocnn_loss(nu, w, V)})
        return model
