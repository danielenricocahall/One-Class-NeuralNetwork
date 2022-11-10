import tensorflow as tf
from flatbuffers.builder import np

from loss import quantile_loss
from ocnn import OneClassNeuralNetwork


def test_loss_function():
    # Test case described in the paper
    # GIVEN we have this data (integers from 1-9) and nu = 1/3 (0.33)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nu = 0.33
    y = tf.convert_to_tensor([data], dtype=tf.float32)
    # WHEN we execute the quantile loss function on each value in the data
    results = {r: quantile_loss(r, y, nu) for r in data}
    # THEN the argument which gives us the minimum value should be 3
    minimum_argument, minimum_value = next((k, v) for k, v in results.items() if v == min(results.values()))
    assert minimum_argument == 3
    assert np.isclose(minimum_value, -1.99, atol=0.01)


def test_build_model():
    ocnn = OneClassNeuralNetwork(3, 5, 1.0)
    model, V, w = ocnn.build_model()
    assert len(model.layers) == 4
    assert V.shape == (3, 5)
    assert w.shape == (5, 1)


