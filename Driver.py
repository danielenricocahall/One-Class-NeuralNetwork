'''
Created on Feb 10, 2019

@author: daniel
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from OC_NN import OC_NN
from sklearn.decomposition import PCA


def main():
    data = h5py.File('http.mat', 'r')
    X = np.array(data['X'], dtype = np.float32).T
    y = np.array(data['y']).T
    y  = 1 - y
    
    num_features = X.shape[1]
    num_hidden = 32
    r = 1.0
    
    epochs = 50
    nu = 0.01
    oc_nn = OC_NN(num_features, num_hidden, r)
    model = oc_nn.trainModel(X, y, epochs, nu)
    y_pred = model.predict(X)
    
    colors = []
    cmap = {0:"b", 1:"r"}

    for i in range(len(y_pred)):
        label = np.rint(y_pred[i,0])
        colors.append(cmap[label])
        
    
    plt.scatter(X[:,0], X[:,1], c = colors)
    plt.show()
 
    
    
    

    
    
if __name__ == "__main__":
    main()
    exit()
    
    