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

    def custom_ocnn_loss(self, nu):
        def custom_hinge(_, y_pred):
            loss = 0.5 * tf.norm(self.w) + \
                   0.5 * tf.norm(self.V) + \
                   quantile_loss(self.r, y_pred, nu)
            self.r = tfp.stats.percentile(y_pred, q=100 * nu, interpolation='linear')
            return loss

        return custom_hinge

    def build_model(self):
        h_size = self.hidden_size
        model = Sequential()
        input_hidden = Dense(h_size, input_dim=self.input_dim, kernel_initializer="glorot_normal", name="input_hidden")
        model.add(input_hidden)
        model.add(Activation("sigmoid"))

        # Define Dense layer from hidden to output
        hidden_output = Dense(1, name="hidden_output")
        model.add(hidden_output)
        model.add(Activation("linear"))

        self.V = input_hidden.get_weights()[0]  # "V is the weight matrix from input to hidden units"
        self.w = hidden_output.get_weights()[0]  # "w is the scalar output obtained from the hidden to output layer"

        return model

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

        def w_norm(*args):
            return 0.5 * tf.norm(self.w)

        def V_norm(*args):
            return 0.5 * tf.norm(self.V)

        w_norm.__name__ = 'w_norm'

        V_norm.__name__ = 'V_norm'

        def quantile_loss_metric(*args):
            return quantile_loss(self.r, args[1], nu)

        def on_epoch_end(epoch, logs):
            self.w = model.get_layer('hidden_output').get_weights()[0]
            self.V = model.get_layer('input_hidden').get_weights()[0]

        quantile_loss_metric.__name__ = 'quantile_loss'

        model = self.build_model()

        model.compile(optimizer=Adam(lr=init_lr),
                      loss=self.custom_ocnn_loss(nu),
                      metrics=[r_metric, quantile_loss_metric, w_norm, V_norm], run_eagerly=True)

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        history = model.fit(X, np.zeros((X.shape[0],)),
                            steps_per_epoch=1,
                            shuffle=True,
                            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)],
                            epochs=epochs)

        if save:
            import os
            from datetime import datetime
            if not os.path.exists('models'):
                os.mkdir('models')
            model_dir = f"models/ocnn_{datetime.now().strftime('%Y-%m-%d-%H:%M:%s')}"
            os.mkdir(model_dir)
            V = model.layers[0].get_weights()[0]
            w = model.layers[2].get_weights()[0]
            model.save(f"{model_dir}/model.h5")
            np.savez(f"{model_dir}/params.npz", w=w, V=V, nu=nu)

        return model, history

    def load_model(self, model_dir):
        """
        loads a pretrained model
        :param model_dir: directory where model and model params (w, V, and nu) are saved
        :return: loaded model
        """
        params = np.load(f'{model_dir}/params.npz')
        w = params['w']
        V = params['V']
        nu = params['nu'].tolist()
        model = load_model(f'{model_dir}/model.h5',
                           custom_objects={'custom_hinge': self.custom_ocnn_loss(nu, w, V)})
        return model
