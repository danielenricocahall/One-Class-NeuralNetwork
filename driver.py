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
    num_hidden = 64
    r = 1.0
    epochs = 300
    nu = 0.001

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(X, epochs=epochs, nu=nu)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="train_loss")
    plt.plot(history.epoch, history.history["quantile_loss"], label="quantile_loss")
    plt.plot(history.epoch, history.history["r"], label="r")

    plt.title("OCNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

    y_pred = model.predict(X)

    r = history.history['r'].pop()

    s_n = [y_pred[i, 0] - r >= 0 for i in range(len(y_pred))]

    frac_of_outliers = len([s for s in s_n if s == 0]) / len(s_n)

    cmap = ListedColormap(['r', 'b'])

    # choose features to use for scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=s_n, cmap=cmap)
    ax.set_xlabel(feature_index_to_name[0])
    ax.set_ylabel(feature_index_to_name[1])
    ax.set_zlabel(feature_index_to_name[2])
    plt.legend(handles=scatter.legend_elements()[0], labels=['anomalous', 'normal'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
    exit()
