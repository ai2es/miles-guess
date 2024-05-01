import keras
import keras.layers as layers
import keras.ops as ops
from keras.src import activations
from keras.src.layers.layer import Layer

@keras.saving.register_keras_serializable()
class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit activation layer.

    This layer allows a small gradient when the unit is not active.

    Formula:

    ``` python
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    ```

    Example:

    ``` python
    leaky_relu_layer = LeakyReLU(negative_slope=0.5)
    input = np.array([-10, -5, 0.0, 5, 10])
    result = leaky_relu_layer(input)
    # result = [-5. , -2.5,  0. ,  5. , 10.]
    ```

    Args:
        negative_slope: Float >= 0.0. Negative slope coefficient.
          Defaults to `0.3`.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.

    """

    def __init__(self, negative_slope=0.3, **kwargs):
        if "alpha" in kwargs:
            negative_slope = kwargs.pop("alpha")
            warnings.warn(
                "Argument `alpha` is deprecated. "
                "Use `negative_slope` instead."
            )
        super().__init__(**kwargs)
        if negative_slope is None or negative_slope < 0:
            raise ValueError(
                "The negative_slope value of a Leaky ReLU layer "
                "cannot be None or negative value. Expected a float."
                f" Received: negative_slope={negative_slope}"
            )
        self.supports_masking = True
        self.negative_slope = negative_slope

    def call(self, inputs):
        return activations.leaky_relu(
            inputs, negative_slope=self.negative_slope
        )

    def get_config(self):
        config = super().get_config()
        config.update({"negative_slope": self.negative_slope})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.saving.register_keras_serializable()
class DenseNormalGamma(layers.Layer):
    """
    Implements dense output layer for a deep evidential regression model.
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning

    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        spectral_normalization (bool): Whether to use Spectral Normalization
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        [mu, lambda, alpha, beta] (see reference documentation for more)
    """
    NUM_OUTPUT_PARAMS = 4

    def __init__(self, units: int,
                 spectral_normalization: bool = False,
                 eps: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(layers.Dense(DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units,
                                                                   activation=None))
        else:
            self.dense = layers.Dense(DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units, activation=None)
        self.eps = eps

    def evidence(self, x):
        """
        Converts values from continuous space to greater than 0 using a softplus activation function.

        Args:
            x: input value

        Returns:
            Tranformed values
        """
        return ops.maximum(ops.softplus(x), self.eps)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = ops.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return ops.concatenate([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config
      

class DenseNormal(layers.Layer):
    """
    Dense output layer for a Gaussian distribution regression neural network.

    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        spectral_normalization (bool): Whether to use Spectral Normalization
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        mu, sigma: mean and standard deviation of output
    """
    NUM_OUTPUT_PARAMS = 2

    def __init__(self, units: int,
                 spectral_normalization: bool = False,
                 eps: float = 1e-12, output_activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.output_activation = output_activation
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(layers.Dense(DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                                                                   activation=self.output_activation))
        else:
            self.dense = layers.Dense(DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                                      activation=self.output_activation)
        self.eps = eps

    def call(self, x):
        output = self.dense(x)
        output = ops.maximum(output, self.eps)
        mu, sigma = ops.split(output, 2, axis=-1)
        return ops.concatenate([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], DenseNormal.NUM_OUTPUT_PARAMS * self.units

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        base_config['eps'] = self.eps
        base_config["output_activation"] = self.output_activation
        return base_config