'''
Created on Mar 21, 2019

@author: daniel
'''
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf
sess = tf.Session()


from keras import backend as K
K.set_session(sess)
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class OC_NN:

    def __init__(self, inputdim, hiddenLayerSize, r = 1.0):
        self.input_dim = inputdim
        self.hidden_size = hiddenLayerSize
        self.r = r

    def custom_ocnn_loss(self,nu, w, V):

        def custom_hinge(_, y_pred):
            
            y = self.r
            y_hat = y_pred
            # r = nuth quantile
            loss = 0.5 * tf.reduce_sum(w ** 2) + 0.5 * tf.reduce_sum(V ** 2) + \
                (1 / nu) * K.mean(K.maximum(0.0, y - y_hat)) - self.r
            self.r = tf.contrib.distributions.percentile(self.r, q = 100 * nu)
            return loss
        return custom_hinge
    
    def buildModel(self, classes):
    
        h_size = self.hidden_size
        model = Sequential()
        input_hidden = Dense(h_size, input_dim = self.input_dim, kernel_initializer="glorot_normal",name="input_hidden")
        model.add(input_hidden)
        model.add(Activation("linear"))
    
        ## Define Dense layer from hidden  to output
        hidden_ouput = Dense(classes,name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation("sigmoid"))
    
    
        with sess.as_default():
            w = input_hidden.get_weights()[0]
            V = hidden_ouput.get_weights()[0]
    
        return [model,w,V]
    
    def trainModel(self, X, y, epochs, nu):
        [model, w, V] = self.buildModel(1)
        init_lr = 1e-2
        model.compile(optimizer = Adam(lr = init_lr, decay = init_lr / epochs), 
                      loss = self.custom_ocnn_loss(nu, w, V))
        
        H = model.fit(X, y, 
                  steps_per_epoch=1, 
                  shuffle = True, 
                  epochs = epochs)
        
        with sess.as_default():
            w = model.layers[0].get_weights()[0]
            V = model.layers[2].get_weights()[0]
        
        plt.style.use("ggplot")
        plt.figure()
        N = epochs
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.title("OC_NN Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()
        
        return model
        
        
    
