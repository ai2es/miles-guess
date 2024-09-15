from mlguess.keras.layers import DenseNormal, DenseNormalGamma
import keras
import numpy as np

def test_DenseNormalGamma():
    x = DenseNormalGamma(1)
    out = keras.ops.convert_to_numpy(x(np.array([[5.5]])))
    assert out.shape[1] == 4
    assert np.min(out[:, 1:]) >= 0

def test_NormalGamma():
    x = DenseNormal(1)
    out = keras.ops.convert_to_numpy(x(np.array([[5.5]])))
    assert out.shape[1] == 2
    assert np.min(out) >= 0
