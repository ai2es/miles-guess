import tensorflow as tf
import numpy as np
import logging


class DirichletEvidentialLoss(tf.keras.losses.Loss):
    """
    Loss function for an evidential categorical model.
    Args:
        callback (list): List of callbacks.
        name (str): reference name
        this_epoch_num (int):  Epoch callback
        class_weights (list): List of class weights (experimental)
    """
    def __init__(self, callback=None, name="dirichlet", this_epoch_num=None, class_weights=None):

        super().__init__()
        self.callback = callback
        self.__name__ = name
        self.class_weights = class_weights
        self.this_epoch_num = this_epoch_num
        if self.class_weights:
            logging.warning("The application of class weights to this loss is experimental.")

    def kl(self, alpha):
        beta = tf.constant(np.ones((1, alpha.shape[1])), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=1, keepdims=True
        )
        lnB_uni = tf.reduce_sum(
            tf.math.lgamma(beta), axis=1, keepdims=True
        ) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        if self.class_weights:
            kl = (tf.reduce_sum(self.class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB +
                  lnB_uni)
        else:
            kl = (tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni)
        return kl

    def __call__(self, y, output, sample_weight=None):
        evidence = tf.nn.relu(output)
        alpha = evidence + 1

        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        if self.class_weights:
            A = tf.reduce_sum(self.class_weights * (y - m) ** 2, axis=1, keepdims=True)
            B = tf.reduce_sum(self.class_weights * alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
        else:
            A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
            B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)

        annealing_coef = tf.minimum(1.0, self.this_epoch_num / self.callback.annealing_coef)
        alpha_hat = y + (1 - y) * alpha
        C = annealing_coef * self.KL(alpha_hat)
        C = tf.reduce_mean(C, axis=1)

        return tf.reduce_mean(A + B + C)


class DirichletEvidentialLoss_keras3(tf.keras.losses.Loss):
    """
    Loss function for an evidential categorical model.
    Args:
        callback (list): List of callbacks.
        name (str): reference name
        this_epoch_num (int):  Epoch callback
        class_weights (list): List of class weights (experimental)
    """
    def __init__(self, callback=None, name="dirichlet", this_epoch_num=None, class_weights=None):

        super().__init__()
        self.callback = callback
        self.__name__ = name
        self.class_weights = class_weights
        self.this_epoch_num = this_epoch_num
        if self.class_weights:
            logging.warning("The application of class weights to this loss is experimental.")

    def kl(self, alpha):
        beta = tf.constant(np.ones((1, alpha.shape[1])), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=1, keepdims=True
        )
        lnB_uni = tf.reduce_sum(
            tf.math.lgamma(beta), axis=1, keepdims=True
        ) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        if self.class_weights:
            kl = (tf.reduce_sum(self.class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB +
                  lnB_uni)
        else:
            kl = (tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni)
        return kl

    def __call__(self, y, output, sample_weight=None):
        evidence = tf.nn.relu(output)
        alpha = evidence + 1

        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        if self.class_weights:
            A = tf.reduce_sum(self.class_weights * (y - m) ** 2, axis=1, keepdims=True)
            B = tf.reduce_sum(self.class_weights * alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
        else:
            A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
            B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)

        annealing_coef = tf.minimum(1.0, self.this_epoch_num / self.callback.annealing_coef)
        alpha_hat = y + (1 - y) * alpha
        C = annealing_coef * self.KL(alpha_hat)
        C = tf.reduce_mean(C, axis=1)

        return tf.reduce_mean(A + B + C)



class EvidentialRegressionLoss(tf.keras.losses.Loss):
    """
    Loss function for an evidential regression model. The total loss is the Negative Log Likelihood of the
    Normal Inverse Gamma summed with the error and scaled by the evidential coefficient. The coefficient has a strong
    influence on the uncertainty predictions (less so for the predictions themselves) of the model and must be tuned
    for individual datasets.
    Loss = loss_nll + coeff * loss_reg
    Args:
        coeff (float): Evidential Coefficient
    """
    def __init__(self, coeff=1.0):
        super(EvidentialRegressionLoss, self).__init__()
        self.coeff = coeff

    def nig_nll(self, y, gamma, v, alpha, beta, reduce=True):
        v = tf.math.maximum(v, tf.keras.backend.epsilon())
        twoBlambda = 2 * beta * (1 + v)
        nll = (0.5 * tf.math.log(np.pi / v)
               - alpha * tf.math.log(twoBlambda)
               + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
               + tf.math.lgamma(alpha)
               - tf.math.lgamma(alpha + 0.5))

        return tf.reduce_mean(nll) if reduce else nll

    def nig_reg(self, y, gamma, v, alpha, reduce=True):
        error = tf.abs(y - gamma)
        evi = 2 * v + alpha
        reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
        loss_nll = self.nig_nll(y_true, gamma, v, alpha, beta)
        loss_reg = self.nig_reg(y_true, gamma, v, alpha)

        return loss_nll + self.coeff * loss_reg

    def get_config(self):
        config = super(EvidentialRegressionLoss, self).get_config()
        config.update({"coeff": self.coeff})
        return config


def gaussian_nll(y, y_pred, reduce=True):
    """
    Loss function for a parametric Gaussian Loss.
    Args:
        y: Training data targets
        y_pred: Model predicitons
    Returns:
        Mean Negative log likelihood
    """
    ax = list(range(1, len(y.shape)))
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    logprob = (-tf.math.log(sigma)
               - 0.5 * tf.math.log(2 * np.pi)
               - 0.5 * ((y - mu) / sigma) ** 2)
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss


class EvidentialRegressionCoupledLoss(tf.keras.losses.Loss):
    def __init__(self, r=1.0, coeff=1.0):
        """
        implementation of the loss from meinert and lavin that fixes issues with the original
        evidential loss for regression. The loss couples the virtual evidence values with coefficient r.
        In this new loss, the regularizer is unnecessary.
        """
        super(EvidentialRegressionCoupledLoss, self).__init__()
        self.coeff = coeff
        self.r = r

    def nig_nll(self, y, gamma, v, alpha, beta, reduce=True):
        # couple the parameters as per meinert and lavin

        twoBlambda = 2 * beta * (1 + v)
        nll = (0.5 * tf.math.log(np.pi / v)
               - alpha * tf.math.log(twoBlambda)
               + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
               + tf.math.lgamma(alpha)
               - tf.math.lgamma(alpha + 0.5))

        return tf.reduce_mean(nll) if reduce else nll

    def nig_reg(self, y, gamma, v, alpha, reduce=True):
        error = tf.abs(y - gamma)  # can try squared loss here to target the right minimizer
        evi = (v + 2 * alpha)  # new paper: = v + 2 * alpha, can try to change this to just 2alpha
        reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
        v = (2 * (alpha - 1) / self.r)  # need to couple this way otherwise alpha could be negative

        loss_nll = self.nig_nll(y_true, gamma, v, alpha, beta)
        loss_reg = self.nig_reg(y_true, gamma, v, alpha)

        return loss_nll + self.coeff * loss_reg

    def get_config(self):
        config = super(EvidentialRegressionCoupledLoss, self).get_config()
        config.update({"r": self.r, "coeff": self.coeff})
        return config
