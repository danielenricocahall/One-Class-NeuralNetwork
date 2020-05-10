import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ocnn import OneClassNeuralNetwork


def main():
    data = h5py.File('Data/http.mat', 'r')
    X = np.array(data['X'], dtype=np.float32).T

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

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(X, epochs=epochs, nu=nu)

    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.title("OCNN Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

    y_pred = model.predict(X)

    y_pred = [np.rint(y_pred[i, 0]) for i in range(len(y_pred))]

    cmap = ListedColormap(['r', 'b'])

    # choose features to use for scatter plot
    i, j = 0, 1

    scatter = plt.scatter(X[:, i], X[:, j], c=y_pred, cmap=cmap)
    plt.xlabel(feature_index_to_name[i])
    plt.ylabel(feature_index_to_name[j])
    plt.legend(handles=scatter.legend_elements()[0], labels=['anomalous', 'normal'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
    exit()
