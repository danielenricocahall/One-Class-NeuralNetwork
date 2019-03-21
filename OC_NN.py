'''
Created on Mar 21, 2019

@author: daniel
'''

# import the necessary packages
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf
sess = tf.Session()
import keras


from keras import backend as K
K.set_session(sess)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
# set the matplotlib backend so figures can be saved in the background
from keras.callbacks import LambdaCallback
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,Adagrad
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

class OC_NN:

    def __init__(self, inputdim, hiddenLayerSize, r = 1.0):
        self.input_dim = inputdim
        self.hidden_size = hiddenLayerSize
        self.r = r

    def custom_ocnn_loss(self,nu, w, V):

        def custom_hinge(_, y_pred):
    
            term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
            term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
            term3 = 1 / nu * K.mean(K.maximum(0.0, self.r - tf.reduce_max(y_pred, axis=1)), axis=-1)
            term4 = -1*self.r
            # yhat assigned to r
            self.r = tf.reduce_max(y_pred, axis=1)
            # r = nuth quantile
            self.r = tf.contrib.distributions.percentile(self.r, q = 100 * nu)
            rval = tf.reduce_max(y_pred, axis=1)
            rval = tf.Print(rval, [tf.shape(rval)])
            return (term1 + term2 + term3 + term4)

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
                  #callbacks = callbacks,
                  epochs = epochs)
        
        with sess.as_default():
            w = model.layers[0].get_weights()[0]
            V = model.layers[2].get_weights()[0]
            #np.save(self.directory+"w", w)
            #np.save(self.directory +"V", V)
                # print("[INFO] ",type(w) ,w.shape,"type of w...")
        
        plt.style.use("ggplot")
        plt.figure()
        N = epochs
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.title("OC_NN Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Vs Epochs")
        plt.legend(loc="upper right")
        plt.show()
        
        return model
        
        
    