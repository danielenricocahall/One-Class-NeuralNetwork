'''
Created on Feb 10, 2019

@author: daniel
'''
      
      
import numpy as np
import tensorflow as tf
import keras.backend as K

g   = lambda x : 1/(1 + tf.exp(-x))
  
    
def OneClassLoss(nu, w, V):
    def custom_hinge(y_true, y_pred):

        term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
        term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
        term3 = 1 / nu * K.mean(K.maximum(0.0, self.r - tf.reduce_max(y_pred, axis=1)), axis=-1)
        term4 = -1*self.r
        # yhat assigned to r
        self.r = tf.reduce_max(y_pred, axis=1)
        # r = nuth quantile
        self.r = tf.contrib.distributions.percentile(self.r, q=100 * nu)
        rval = tf.reduce_max(y_pred, axis=1)
        rval = tf.Print(rval, [tf.shape(rval)])



        return (term1 + term2 + term3 + term4)

    return custom_hinge

