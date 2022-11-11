# One-Class-NeuralNetwork
Simplified Keras implementation of one class neural network for nonlinear anomaly detection. 

The implementation is based on the approach described here: https://arxiv.org/pdf/1802.06360.pdf. I've included several datasets from ODDS (http://odds.cs.stonybrook.edu/) and the Wine Dataset from UCI (https://archive.ics.uci.edu/ml/datasets/wine) to play with.

# Setup

`pipenv install .` should configure a python environment and install all necessary dependencies in the environment. 

# Running

Running `python kdd_cup.py` or `python wine.py` within your new python environment (either through CLI or IDE) should kick off training on the KDD Cup dataset epochs and generate some output plots.

# Testing

Two unit tests are defined in `test/test_basic.py`: building the model, and the quantile loss test based on example in the paper:

![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/test_case.png)

Execute `pytest test` to run.

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


# Limitations
Unlike in the original paper, which optimizes for `w` and `V` first, then optimizes for `r`, in this implementation, they're jointly optimized, which results in an unbounded loss (as in, it never converges). The model works, but I would like to explore this further to evaluate if we can perform the correct optimization in one step.
