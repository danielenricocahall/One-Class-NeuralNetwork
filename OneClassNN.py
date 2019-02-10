'''
Created on Nov 19, 2018

@author: dcahall
'''

## Attempt at a OC-NN implementation, let's see how this goes...


import tensorflow as tf
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import random
from FeatureExtraction import FeatureExtraction

# define sigmoid as our function 'g'
g   = lambda x : 1/(1 + tf.exp(-x))


class OneClassNN():
    model = None;
    X = None;
    r = None
    ## Parameter on the domain (0,1) which controls the precision-recall tradeoff
    def __init__(self, num_inputs, num_hidden, X):
        self.r = 0.1
        self.X = np.array(X, dtype=np.float32)
        self.buildModel(num_inputs, num_hidden)
        
        
        
    def buildModel(self, num_features, num_hidden):
        model = Sequential()
        model.add(Dense(num_hidden, input_dim = num_features, activation = "relu"))
        model.add(Dense(1 ,activation = "sigmoid"))
        self.model = model
        self.model.compile(optimizer='adam',
              loss=self.OneClassLoss(),
              metrics=[self.OneClassLoss()])
        

    

    def fit(self, X):
        #self.X = X
        y = tf.zeros((X.shape[0],))
        X = np.array(X,dtype=np.float32)
        self.model.fit(X,y,steps_per_epoch=1, epochs=500)
        return self.model
    
    
    def OneClassLoss(self):
        nu = 0.5
        X = self.X
        V = np.array(self.model.layers[0].get_weights()[0])
        def loss(_, w):
            r = np.percentile(w,q=100*nu)
            return 0.5*tf.reduce_sum(w**2) + 0.5*tf.reduce_sum(V**2) + \
                 1/nu * tf.reduce_mean(tf.maximum(r - g(tf.matmul(X,V)),0) * w) - r
        

        return loss

        
def main():
    featurelist = np.load('./Models/featurelist.npy')
    N = 1000
    mt_list = np.load('./Models/mt_list.npy')
    _, message_name, _ = zip(*mt_list)
    msg_data=[]
    random.seed(0)
    for m in message_name:
        cumsum = 0
        for _ in range(0,N):
            cumsum = cumsum + random.random()
            msg_data.append([m, cumsum])
    msg_data.sort(key=lambda x:x[1])
    #print(msg_data)
    #If loop over schema, split lines to grow in loop
    fe = FeatureExtraction(False)
    fe.grow_measurements(msg_data)
    
    x_data_df = fe.get_train_feats()
    groups = x_data_df.groupby(['message_name'])
    for name, _ in groups:
        feattrace_df = np.array(groups[featurelist].get_group(name), dtype=np.float32)
        OCNN = OneClassNN(feattrace_df.shape[1], 100, feattrace_df)
        OCNN.fit(feattrace_df)
        
if __name__=="__main__":
    main()
        
        
        


        
        
        