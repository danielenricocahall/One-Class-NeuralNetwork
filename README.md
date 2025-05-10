# One-Class-NeuralNetwork
Simplified Keras implementation of one class neural network for nonlinear anomaly detection. 

The implementation is based on the approach described here: https://arxiv.org/pdf/1802.06360.pdf. I've included several datasets from ODDS (http://odds.cs.stonybrook.edu/) and the Wine Dataset from UCI (https://archive.ics.uci.edu/ml/datasets/wine) to play with.

# Setup

`uv sync` should configure a python environment and install all necessary dependencies in the environment. 

# Running

Running `uv run python kdd_cup.py` or `uv run python wine.py` within your new python environment (either through CLI or IDE) should kick off training on the KDD Cup dataset epochs and generate some output plots.

# Testing

Two unit tests are defined in `test/test_basic.py`: building the model, and the quantile loss test based on example in the paper:

![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/test_case.png)

Execute `uv run pytest test` to run.

# Results

## HTTP Dataset ##

### Loss ###

![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/loss_http.png)


### Features ###
![alt_text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/feat_http.png)


## Wine Dataset ###

## Loss ##
![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/wine_loss.png)


## Features ##
![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/wine_clusters.png)


# Notes
Based on the objective function we use, the loss is unbounded, meaning there is no real "convergence" - at least on the two test datasets presented here. I've probed the source code, read through the paper several times, and I'm fairly certain the implementation here is accurate. I'm not sure if this is a limitation of the approach, or if I'm missing something. I'd love to hear from anyone who has any insight on this, especially if you apply it to new datasets.
