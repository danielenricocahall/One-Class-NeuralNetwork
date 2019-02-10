'''
Created on Feb 10, 2019

@author: daniel
'''
      
      
import numpy as np
import tensorflow as tf

g   = lambda x : 1/(1 + tf.exp(-x))
  
    
def OneClassLoss(nu, X, model):
    V = np.array(model.layers[0].get_weights()[0])
    def loss(_, w):
        r = np.percentile(w,q=100*nu)
        return 0.5*tf.reduce_sum(w**2) + 0.5*tf.reduce_sum(V**2) + \
             1/nu * tf.reduce_mean(tf.maximum(r - g(tf.matmul(X,V)),0) * w) - r
    return loss
