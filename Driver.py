'''
Created on Feb 10, 2019

@author: daniel
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from OC_NN import OC_NN


def main():
    data = h5py.File('http.mat', 'r')
    X = np.array(data['X'], dtype = np.float32).T
    y = np.array(data['y']).T
    y = np.array(y)
    print(np.count_nonzero(y) / y.shape[0])
    y = 1 - y
    num_features = X.shape[1]
    num_hidden = 32
    r = 1.0
    
    epochs = 100
    nu = 0.99
    oc_nn = OC_NN(num_features, num_hidden, r)
    model = oc_nn.trainModel(X, y, epochs, nu)
    y_pred = model.predict(X)
    plt.plot(np.arange(len(y_pred)), y_pred, '*')
    plt.plot(np.arange(len(y)), y)
    plt.show()
    

    
    
if __name__ == "__main__":
    main()
    exit()
    
    