'''
Created on Feb 10, 2019

@author: daniel
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from OC_NN import OC_NN


def main():
    data = h5py.File('Data/http.mat', 'r')
    X = np.array(data['X'], dtype = np.float32).T
    y = np.array(data['y']).T
    y  = 1 - y
    
    ## 1 = normal
    ## 0 = anomalous
    
    """
    Mapping derived from http://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/ and http://odds.cs.stonybrook.edu/http-kddcup99-dataset/
    """
    feature_index_to_name = {0: "duration",
                             1: "src_bytes",
                             2: "dst_bytes"}
    

    
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
        
    
    ## choose features to use for scatter plot
    i, j = 0, 1

    plt.scatter(X[:,i], X[:,j], c = colors)
    plt.xlabel(feature_index_to_name[i])
    plt.ylabel(feature_index_to_name[j])
    plt.show()
 
    
    
    

    
    
if __name__ == "__main__":
    main()
    exit()
    
    