import os
import sys
import keras
import keras.ops as ops
import numpy as np
from keras.regularizers import L1, L2, L1L2
from keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from mlguess.keras.layers import DenseNormalGamma
import logging

class CategoricalDNN(keras.models.Model):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
        classifier: (boolean) If training on classes
    """
    def __init__(self, hidden_layers=2, hidden_neurons=64, activation="relu", output_activation="softmax",
                 optimizer="adam", loss="categorical_crossentropy",loss_weights=None, annealing_coeff=1.0,
                 use_noise=False, noise_sd=0.0, lr=0.001, use_dropout=False, dropout_alpha=0.2, batch_size=128,
                 epochs=2, kernel_reg=None, l1_weight=0.0, l2_weight=0.0, sgd_momentum=0.9, adam_beta_1=0.9,
                 adam_beta_2=0.999, epsilon=1e-7, decay=0, verbose=0, random_state=1000, callbacks=[],
                 balanced_classes=0,steps_per_epoch=0, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.annealing_coeff = annealing_coeff
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.callbacks = callbacks
        self.decay = decay
        self.verbose = verbose
        self.random_state = random_state
        self.n_classes = n_classes
        self.hyperparameters = ["hidden_layers", "hidden_neurons", "activation", "output_activation",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon", "loss",
                                "loss_weights", "annealing_coeff", "lr", "kernel_reg", "l1_weight", "l2_weight",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "callbacks", "decay", "verbose", "random_state", "n_classes"]
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        if self.activation == "leaky":
            self.activation = LeakyReLU()
        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None
        self.model_layers = []
        for h in range(self.hidden_layers):
            self.model_layers.append(Dense(self.hidden_neurons,
                                                       activation=self.activation,
                                                       kernel_regularizer=self.kernel_reg,
                                                       name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model_layers.append(Dense(self.n_classes, activation=self.output_activation, name="dense_output"))

    def call(self, inputs):

        layer_output = self.model_layers[0](inputs)

        for l in range(1, len(self.model_layers)):
            layer_output = self.model_layers[l](layer_output)

        return layer_output

    def fit(self, x, y, **kwargs):

        hist = super().fit(x, y, **kwargs)

        return hist

    def predict(self, x, return_uncertainties=True, **kwargs):

        output = super().predict(x, **kwargs)
        if return_uncertainties:
            return self.calc_uncertainty(output)
        elif return_uncertainties == False:
            return output
        else:
            raise ValueError("return_uncertainties must be a bool")

    def calc_uncertainty(self, y_pred):
        num_classes = y_pred.shape[-1]
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        S = ops.sum(alpha, axis=1, keepdims=True)
        u = num_classes / S
        prob = alpha / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic

     def get_config(self):

            base_config = super().get_config()
            parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
            return {**base_config, **parameter_config}

class EvidentialRegressorDNN(keras.models.Model):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and provides evidential uncertainty estimation.
    Inherits from BaseRegressor.

    Attributes:
        hidden_layers: Number of hidden layers.
        hidden_neurons: Number of neurons in each hidden layer.
        activation: Type of activation function.
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object.
        use_noise: Whether additive Gaussian noise layers are included in the network.
        noise_sd: The standard deviation of the Gaussian noise layers.
        use_dropout: Whether Dropout layers are added to the network.
        dropout_alpha: Proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch.
        epochs: Number of epochs to train.
        verbose: Level of detail to provide during training.
        model: Keras Model object.
        evidential_coef: Evidential regularization coefficient.
        metrics: Optional list of metrics to monitor during training.
    """
    def __init__(self, hidden_layers=2, hidden_neurons=64, activation="relu", optimizer="adam", loss_weights=None,
                 use_noise=False, noise_sd=0.01, lr=0.00001, use_dropout=False, dropout_alpha=0.1, batch_size=128,
                 epochs=2, kernel_reg=None, l1_weight=0.01, l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9,
                 adam_beta_2=0.999, epsilon=1e-7, verbose=1, **kwargs):

        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss_weights = loss_weights
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer_obj = None
        self.training_var = None
        self.epsilon = epsilon
        self.n_output_tasks = 1
        self.N_OUTPUT_PARAMS = 4
        self.hyperparameters = ["hidden_layers", "hidden_neurons", "activation", "training_var",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon", "loss",
                                "loss_weights", "annealing_coeff", "lr", "kernel_reg", "l1_weight", "l2_weight",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "callbacks", "decay", "verbose", "random_state", "n_classes"]

        if self.activation == "leaky":
            self.activation = LeakyReLU()
        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None
        self.model_layers = []
        for h in range(self.hidden_layers):
            self.model_layers.append(Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model_layers.append(DenseNormalGamma(self.n_output_tasks, name="dense_output"))

    def call(self, inputs):

        layer_output = self.model_layers[0](inputs)

        for l in range(1, len(self.model_layers)):
            layer_output = self.model_layers[l](layer_output)

        return layer_output

    def fit(self, x, y, **kwargs):

        hist = super().fit(x, y, **kwargs)
        self.training_var = np.var(x, axis=-1)

        return hist

    def predict(self, x, return_uncertainties=True, batch_size=10000):
        """
        Args:
            x: Input data
            batch_size: Size of batch to predict
            return_uncertainties: Returns derived uncertainties from evidential distribution parameters.
                                  If False, return the raw parameters themselves (mu, gamma, alpha, beta).
        Returns:
            If return_uncertainties is True: np.array(mu, aleatoric uncertainty, epistemic uncertainty)
            Else If return_uncertainties is False: np.array(mu, gamma, alpha, beta)
        """
        output = super().predict(x, batch_size=batch_size)
        if return_uncertainties:
            return self.calc_uncertainties(output)
        elif return_uncertainties == False:
            return output
        else:
            raise ValueError("return_uncertainties must be a bool")

    def calc_uncertainties(self, preds):

        mu, v, alpha, beta = np.split(preds, self.N_OUTPUT_PARAMS, axis=-1)

        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))

        if len(mu.shape) == 1:
            mu = np.expand_dims(mu, 1)
            aleatoric = np.expand_dims(aleatoric, 1)
            epistemic = np.expand_dims(epistemic, 1)

        for i in range(mu.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
            epistemic[:, i] *= self.training_var[i]

        return np.concatenate([mu, aleatoric, epistemic], axis=-1)

    def get_config(self):

        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}


