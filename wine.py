import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ocnn import OneClassNeuralNetwork


def main():
    # based on mapping from https://archive.ics.uci.edu/ml/datasets/wine
    flavanoid_index = 7
    color_index = 10
    with open('data/wine.data', 'r') as fp:
        data = fp.readlines()
        data = [line.strip().split(',') for line in data]
        data = [[float(x) for x in line] for line in data]
        data = [[line[flavanoid_index], line[color_index]] for line in data]
    X = np.array(data, dtype=np.float32)

    feature_index_to_name = {0: "Concentration of flavanoids",
                             1: "Color intensity"}

    num_features = X.shape[1]
    num_hidden = 8
    r = 1.0
    epochs = 500
    nu = 0.05

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(X, epochs=epochs, nu=nu, init_lr=0.01)

    plt.style.use("ggplot")
    plt.figure()
    # Note: omit the first train loss as it is very high and skews the plot
    plt.plot(history.epoch[1:], history.history["loss"][1:], label="train_loss")
    plt.plot(history.epoch, history.history["quantile_loss"], label="quantile_loss")
    plt.plot(history.epoch, history.history["r"], label="r")
    plt.plot(history.epoch, history.history["w_norm"], label="w_norm")
    plt.plot(history.epoch, history.history["V_norm"], label="V_norm")

    plt.title("OCNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("figures/wine_loss.png", dpi=110, bbox_inches="tight")
    plt.show()

    y_pred = model.predict(X)

    # solve for r once more against the final (w, V) - the last value recorded in `history` was
    # the one used during the final epoch, before that epoch's gradient step
    oc_nn.update_r(X, nu)
    r = oc_nn.r.numpy()

    s_n = [y_pred[i, 0] - r >= 0 for i in range(len(y_pred))]

    print(f"r = {r:.4f}, fraction flagged anomalous = {1 - sum(s_n) / len(s_n):.4f} (nu = {nu})")

    cmap = ListedColormap(['r', 'b'])

    # choose features to use for scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=s_n, cmap=cmap)
    ax.set_xlabel(feature_index_to_name[0])
    ax.set_ylabel(feature_index_to_name[1])
    plt.legend(handles=scatter.legend_elements()[0], labels=['anomalous', 'normal'], loc='upper right')
    plt.savefig("figures/wine_clusters.png", dpi=110, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
