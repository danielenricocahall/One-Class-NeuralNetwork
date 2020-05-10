import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sess = tf.Session()
K.set_session(sess)


class OneClassNeuralNetwork:

    def __init__(self, input_dim, hidden_layer_size, r=1.0):
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = r

    def custom_ocnn_loss(self, nu, w, V):
        def custom_hinge(_, y_pred):
            y = self.r
            y_hat = y_pred
            # r = nuth quantile
            loss = 0.5 * tf.reduce_sum(w ** 2) + 0.5 * tf.reduce_sum(V ** 2) + \
                   (1 / nu) * K.mean(K.maximum(0.0, y - y_hat)) - self.r
            self.r = tf.contrib.distributions.percentile(self.r, q=100 * nu)
            return loss

        return custom_hinge

    def build_model(self):
        h_size = self.hidden_size
        model = Sequential()
        input_hidden = Dense(h_size, input_dim=self.input_dim, kernel_initializer="glorot_normal", name="input_hidden")
        model.add(input_hidden)
        model.add(Activation("linear"))

        # Define Dense layer from hidden to output
        hidden_ouput = Dense(1, name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation("sigmoid"))

        with sess.as_default():
            w = input_hidden.get_weights()[0]
            V = hidden_ouput.get_weights()[0]

        return [model, w, V]

    def train_model(self, X, epochs, nu, init_lr=1e-2, save=True):
        """

        :param X: training data
        :param epochs: number of epochs to train for
        :param nu:
        :param init_lr: initial learning rate
        :param save: flag indicating if the model should be  (default True)
        :return: trained model
        """
        [model, w, V] = self.build_model()
        model.compile(optimizer=Adam(lr=init_lr, decay=init_lr / epochs),
                      loss=self.custom_ocnn_loss(nu, w, V))

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        result = model.fit(X, np.zeros((X.shape[0], )),
                           steps_per_epoch=1,
                           shuffle=True,
                           epochs=epochs)

        if save:
            import os
            from datetime import datetime
            if not os.path.exists('models'):
                os.mkdir('models')
            model.save(f"models/ocnn_{datetime.now().strftime('%Y-%m-%d-%H:%M')}.h5")

        plt.style.use("ggplot")
        plt.figure()
        N = epochs
        plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
        plt.title("OCNN Training Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()

        return model