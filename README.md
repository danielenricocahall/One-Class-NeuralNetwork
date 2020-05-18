# One-Class-NeuralNetwork
Simplified Keras implementation of one class neural network for nonlinear anomaly detection. 

The implementation is based on the approach described here: https://arxiv.org/pdf/1802.06360.pdf. I've included several datasets from ODDS (http://odds.cs.stonybrook.edu/) to play with.

# Setup

`pipenv install .` should configure a python environment and install all necessary dependencies in the environment. 

# Running

Running `python driver.py` within your new python environment (either through CLI or IDE) should kick off training for 50 epochs and generate some output plots.
# Results

## HTTP Dataset ##

### Loss ###

![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/Figures/loss_http.png)


### Features ###
![alt_text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/Figures/feat_http.png)

# Limitations

* Currently limited to `Tensorflow 1.x` - specifically, `tf.contrib` hasn't been converted to 2.x (https://github.com/tensorflow/models/issues/7767), which is used in the hinge loss function.
* Make demo script more flexible
* Add unit testing
