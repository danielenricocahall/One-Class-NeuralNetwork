import tensorflow as tf
from pytest import fixture

from loss import quantile_loss


@fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()


def test_loss_function(tf_session):
    # Test case described in the paper
    # GIVEN we have this data (integers from 1-9) and nu = 1/3 (0.33)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nu = 0.33
    y = tf.convert_to_tensor([data], dtype=tf.float32)
    # WHEN we execute the quantile loss function on each value in the data
    results = {r: tf_session.run(quantile_loss(r, y, nu)) for r in data}
    # THEN the argument which gives us the minimum value should be 3
    assert next(k for k, v in results.items() if v == min(results.values())) == 3
