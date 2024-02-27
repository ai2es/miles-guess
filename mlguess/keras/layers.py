import tensorflow as tf
import tensorflow_addons as tfa

class DenseNormalGamma(tf.keras.layers.Layer):
    """
    Implements dense layer for an deep evidential regression model.
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning

    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        [mu, lambda, alpha, beta] (see reference documentation for more)
    todo:
        Make Spectral Normalization an option
    """
    
    def __init__(self, units, name, spectral_normalization=False, eps=1e-12, **kwargs):
        """ """
        super(DenseNormalGamma, self).__init__(name=name, **kwargs)
        self.units = int(units)
        if spectral_normalization:
            self.dense = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(4 * self.units, activation=None))
        elif spectral_normalization == False:
            self.dense = tf.keras.layers.Dense(4 * self.units, activation=None)
        self.eps = eps

    def evidence(self, x):
        return tf.math.maximum(tf.nn.softplus(x), self.eps)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4 * self.units

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config
      
        
        
class DenseNormal(tf.keras.layers.Layer):
    """ Dense layer for Gaussian distribution regression.
    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        [mu, sigma] (mean, standard deviation)
    todo:
        Make Spectral Normalization an option
    """
    def __init__(self, units, name, spectral_normalization=False, eps=1e-12):
        super(DenseNormal, self).__init__(name=name)
        self.units = int(units)
        if spectral_normalization:
            self.dense = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(2 * self.units, activation="sigmoid"))
        elif spectral_normalization == False:
            self.dense = tf.keras.layers.Dense(2 * self.units, activation="sigmoid")
        self.eps = eps

    def call(self, x):
        output = self.dense(x)
        output = tf.math.maximum(output, self.eps)
        mu, sigma = tf.split(output, 2, axis=-1)
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config