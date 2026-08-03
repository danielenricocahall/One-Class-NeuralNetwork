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

**Previously this README noted that the loss appeared unbounded and never converged. That turned out to be three bugs in this implementation, not a limitation of the approach.** The objective in Eq (4) of the paper is bounded below and the implementation now converges. For anyone who hit the same wall, the three causes were:

1. **The regularization term received no gradient.** `self.w` and `self.V` were captured via `layer.get_weights()`, which returns *numpy copies*. Inside the loss they were graph constants, so autodiff saw zero gradient from `0.5*||w|| + 0.5*||V||`. The optimizer was minimizing the quantile term alone, with nothing pushing back. Fixed by reading the live layer kernels (`layer.kernel`).

2. **The norms were not squared.** The paper uses `0.5*||w||²₂ + 0.5*||V||²_F`; the code used `tf.norm`, which is unsquared. This is not cosmetic. Scaling `w → c·w` scales the score, `r`, and the whole quantile term linearly in `c`, while the unsquared penalty also grows only linearly. Since the quantile term is negative at its optimum, whenever `|L_q| > 0.5*||w||₂` the objective is **genuinely unbounded below** and gradient descent walks `c → ∞` forever. With the squared norm the penalty grows like `c²`, making the objective coercive and bounded. Fixed in `loss.regularization_loss`.

3. **The `Dense` layers had biases.** The paper scores a point as exactly `⟨w, g(Vx)⟩`, with no bias terms. An output bias is unregularized and shifts the score and `r` together, so the network can drive the objective down by simply making its output constant — which it does, collapsing to `w → 0` and a degenerate solution that separates nothing. Fixing (1) and (2) alone still produced this collapse. Fixed with `use_bias=False`.

Also corrected along the way: `steps_per_epoch=1` meant each epoch trained on a single default-size batch of 32 samples (on `http.mat`, that is ~3,200 of 567,000 rows ever seen, with `r` estimated from 32 points); training is now full-batch, matching the paper's alternating minimization. `RandomNormal(0.5)` was setting the *mean* to 0.5, not the stddev. And `r` is now solved with `interpolation='lower'`, since the empirical minimizer of the pinball loss is the lower quantile — the test case above demonstrates this, with the true minimum at `r=3` (−1.990) versus 3.64 (−1.983) under linear interpolation.

The loss now flattens and stays flat, and `nu` behaves as documented — it controls the fraction of points permitted to cross the hyperplane: Wine flags 4.5% at `nu=0.05`, and HTTP flags 1.00% at `nu=0.01`.

The plots below hold everything else fixed and vary only the regularization term. In both broken variants `w_norm` grows quadratically without bound — exactly the `c → ∞` scaling described above — while the corrected objective settles within a couple hundred epochs and stays flat.

### Wine (8000 epochs) ###
![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/convergence_comparison_wine.png)

### HTTP (1500 epochs) ###
![alt text](https://github.com/danielenricocahall/One-Class-NeuralNetwork/blob/master/figures/convergence_comparison_http.png)

| variant | final loss (wine) | final loss (http) |
| --- | --- | --- |
| unsquared norm + detached weights (original) | -444.8 | -194.0 |
| unsquared norm, gradients connected | -531.1 | -201.2 |
| **squared norm, gradients connected (Eq 4)** | **-1.449** | **-5.358** |
