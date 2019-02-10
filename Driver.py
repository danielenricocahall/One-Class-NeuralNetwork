'''
Created on Feb 10, 2019

@author: daniel
'''

from Loss.OneClassLoss import OneClassLoss
from keras.layers import Dense
from keras.models import Sequential
import h5py
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt

    
def buildModel(num_features, num_hidden):
    model = Sequential()
    model.add(Dense(num_hidden, input_dim = num_features, activation = "relu"))
    model.add(Dense(1 ,activation = "sigmoid"))
    return model


def main():
    nu = 0.004
    data = h5py.File('http.mat', 'r')
    print(list(data.keys()))
    X = np.array(data['X'], dtype = np.float32).T
    y = np.array(data['y']).T
    y = np.array(y)
    num_features = X.shape[1]
    num_hidden = 300
    model = buildModel(num_features, num_hidden)
    loss = OneClassLoss(nu, X, model)
    model.compile(optimizer = Adam(0.01), loss = loss)
    model.fit(X, y, steps_per_epoch=1, epochs = 100)
    y_pred = model.predict(X)
    plt.plot(np.arange(len(y_pred)), y_pred)
    plt.plot(np.arange(len(y)), y)
    plt.show()
    

    
    
if __name__ == "__main__":
    main()
    exit()
    
    