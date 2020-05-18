import tensorflow as tf
from pytest import fixture

from loss import quantile_loss


@fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()


def test_loss_function(tf_session):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_n = tf.convert_to_tensor([data], dtype=tf.float32)
    results = {r: tf_session.run(quantile_loss(r, y_n, 0.33)) for r in data}
    assert next(k for k, v in results.items() if v == min(results.values())) == 3
