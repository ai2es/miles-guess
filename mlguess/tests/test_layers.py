from mlguess.keras.layers import DenseNormal, DenseNormalGamma
import numpy as np

def test_DenseNormalGamma():
    x = DenseNormalGamma(1)
    out = x(np.array([[5.5]])).numpy()
    assert out.shape[1] == 4
    assert np.min(out[:, 1:]) >= 0

def test_NormalGamma():
    x = DenseNormal(1)
    out = x(np.array([[5.5]])).numpy()
    assert out.shape[1] == 2
    assert np.min(out) >= 0
