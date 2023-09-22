import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from mlguess.keras.layers import DenseNormalGamma, DenseNormal
from mlguess.keras.losses import EvidentialRegressionLoss, EvidentialRegressionCoupledLoss, GaussianNLL
from mlguess.keras.losses import DirichletEvidentialLoss
from mlguess.keras.callbacks import ReportEpoch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class RegressorDNN(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        evidential_coef: Evidential regularization coefficient
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        optimizer="adam",
        loss="mse",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
    ):

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss = loss
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
        self.save_path = save_path
        self.model_name = model_name
        self.model = None
        self.optimizer_obj = None
        self.training_std = None
        self.training_var = None
        self.metrics = metrics

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """

        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input

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

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=L2(self.l2_weight),
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )
        nn_model = Dense(outputs, name="dense_last")(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=self.metrics,
            run_eagerly=False,
        )

    def fit(
        self,
        x,
        y,
        validation_data=None,
        callbacks=None,
        initial_epoch=0,
        steps_per_epoch=None,
        workers=1,
        use_multiprocessing=False,
    ):

        self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            callbacks=callbacks,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=True,
        )

        return

    def save_model(self):
        tf.keras.models.save_model(
            self.model, os.path.join(self.save_path, self.model_name), save_format="h5"
        )
        return

    @classmethod
    def load_model(cls, conf):
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logger.info(
            f"Loading a RegressorDNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)
        return model_class

    def predict(self, x, scaler=None, batch_size=None, y_scaler=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        if y_scaler:
            if y_out.shape[-1] == 1:
                y_out = np.expand_dims(y_out, 1)
            y_out = y_scaler.inverse_transform(y_out)
        return y_out

    def predict_monte_carlo(
        self, x_test, y_test, forward_passes, y_scaler=None, batch_size=None
    ):
        _batch_size = self.batch_size if batch_size is None else batch_size
        n_samples = x_test.shape[0]
        pred_size = y_test.shape[1]
        dropout_mu = np.zeros((forward_passes, n_samples, pred_size))

        for i in range(forward_passes):
            output = [
                self.model(x_test[i : i + _batch_size], training=True)
                for i in range(0, x_test.shape[0], _batch_size)
            ]
            output = np.concatenate(output, axis=0)
            if y_scaler:
                if output.shape[-1] == 1:
                    output = np.expand_dims(output, 1)
                output = y_scaler.inverse_transform(output)
            dropout_mu[i] = output
        return dropout_mu
    
    def predict_ensemble(self, x, weight_locations, y_scaler=None, batch_size=None):
        num_models = len(weight_locations)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(weight_locations[0])
            first_prediction = self.predict(x, batch_size=batch_size, y_scaler = y_scaler)
            output_shape = first_prediction.shape[1:]
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)
            predictions[0] = first_prediction
        else:
            output_shape = ()  # Default shape if no models
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(weight_locations[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            y_prob = self.predict(x, batch_size=batch_size, y_scaler=y_scaler)
            predictions[i + 1] = y_prob

        return predictions
    
    

class EvidentialRegressorDNN(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        loss: either evidentialReg (original) or evidentialFix (meinert and lavin)
        coupling_coef: coupling factor for virtual counts in evidentialFix
        evidential_coef: Evidential regularization coefficient
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
        eps: Smallest value of any NN output
    """

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        loss="evidentialReg",
        coupling_coef=1.0,  # right now we have alpha = ... v.. so alpha will be coupled in new loss
        evidential_coef=0.05,
        optimizer="adam",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        uncertainties=True,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
        eps=1e-7,  # smallest eps for stable performance with float32s
    ):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.coupling_coef = coupling_coef
        self.evidential_coef = evidential_coef
        if (
            loss == "evidentialReg"
        ):  # retains backwards compatibility since default without loss arg is original loss
            self.loss = EvidentialRegressionLoss(coeff=self.evidential_coef)
        elif (
            loss == "evidentialFix"
        ):  # by default we do not regularize this loss as per meinert and lavin
            self.loss = EvidentialRegressionCoupledLoss(
                coeff=self.evidential_coef, r=self.coupling_coef
            )
        else:
            raise ValueError("loss needs to be one of evidentialReg or evidentialFix")
            
        logger.info(f"Using loss: {loss}")

        self.uncertainties = uncertainties
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
        self.save_path = save_path
        self.model_name = model_name
        self.model = None
        self.optimizer_obj = None
        self.training_std = None
        self.training_var = None
        self.metrics = metrics
        self.eps = eps

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """

        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input

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

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=L2(self.l2_weight),
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )
        nn_model = DenseNormalGamma(outputs, name="DenseNormalGamma", eps=self.eps)(
            nn_model
        )
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr
            )  # , beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        if self.metrics == "mae":
            metrics = self.mae
        elif self.metrics == "mse":
            metrics = self.mse
        else:
            metrics = None
        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )
        # self.training_var = [np.var(outputs[:, i]) for i in range(outputs)]

    def fit(
        self,
        x,
        y,
        validation_data=None,
        callbacks=None,
        initial_epoch=0,
        steps_per_epoch=None,
        workers=1,
        use_multiprocessing=False,
    ):
        # self.build_neural_network(x.shape[-1], y.shape[-1])
        self.training_var = [np.var(y[:, i]) for i in range(y.shape[-1])]
        
        if self.verbose:
            self.model.summary()
        
        history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            callbacks=callbacks,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=True,
        )

        return history

    def save_model(self):
        # Save the model weights
        tf.keras.models.save_model(
            self.model, os.path.join(self.save_path, self.model_name), save_format="h5"
        )
        # Save the training variances
        np.savetxt(
            os.path.join(self.save_path, f'{self.model_name.strip(".h5")}_training_var.txt'),
            np.array(self.training_var),
        )
        return

    @classmethod
    def load_model(cls, conf):
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logger.info(
            f"Loading an evidential DNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        model_class.training_var = np.loadtxt(
            os.path.join(os.path.join(conf["model"]["save_path"], "best_training_var.txt"))
        )
        
        if not model_class.training_var.shape:
            model_class.training_var = np.array([model_class.training_var])

        return model_class

    def predict(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        if self.uncertainties:
            y_out_final = self.calc_uncertainties(
                y_out, scaler
            )  # todo calc uncertainty for coupled params
        else:
            y_out_final = y_out
        return y_out_final

    def mae(self, y_true, y_pred):
        mu, _, _, _ = tf.split(y_pred, 4, axis=-1)
        return tf.keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        mu, _, _, _ = tf.split(y_pred, 4, axis=-1)
        return tf.keras.metrics.mean_squared_error(y_true, mu)

    def calc_uncertainties(self, preds, y_scaler):
        mu, v, alpha, beta = np.split(preds, 4, axis=-1)

        if isinstance(self.loss, EvidentialRegressionCoupledLoss):
            v = (
                2 * (alpha - 1) / self.coupling_coef
            )  # need to couple this way otherwise alpha could be negative
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))

        if len(mu.shape) == 1:
            mu = np.expand_dims(mu, 1)
            aleatoric = np.expand_dims(aleatoric, 1)
            epistemic = np.expand_dims(epistemic, 1)

        if y_scaler:
            mu = y_scaler.inverse_transform(mu)

        for i in range(mu.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
            epistemic[:, i] *= self.training_var[i]

        return mu, aleatoric, epistemic

    def predict_dist_params(self, x, y_scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        preds = self.model.predict(x, batch_size=_batch_size)
        mu, v, alpha, beta = np.split(preds, 4, axis=-1)
        if isinstance(self.loss, EvidentialRegressionCoupledLoss):
            v = (
                2 * (alpha - 1) / self.coupling_coef
            )  # need to couple this way otherwise alpha could be negative

        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu, 1)
        if y_scaler is not None:
            mu = y_scaler.inverse_transform(mu)

        return mu, v, alpha, beta
    
    def predict_ensemble(self, x, weight_locations, scaler=None, batch_size=None):
        num_models = len(weight_locations)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(weight_locations[0])
            mu, ale, epi = self.predict(x, batch_size=batch_size, scaler=scaler)
            output_shape = mu.shape[1:]
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_ale = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_epi = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_mu[0] = mu
            ensemble_ale[0] = ale
            ensemble_epi[0] = epi
        else:
            output_shape = ()  # Default shape if no models
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_ale = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_epi = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(weight_locations[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            mu, ale, epi =  self.predict(x, batch_size=batch_size, scaler=scaler)
            ensemble_mu[i + 1] = mu
            ensemble_ale[i + 1] = ale
            ensemble_epi[i + 1] = epi

        return ensemble_mu, ensemble_ale, ensemble_epi


class GaussianRegressorDNN(EvidentialRegressorDNN):
    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        self.loss = GaussianNLL

        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input

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

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=L2(self.l2_weight),
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )
        nn_model = DenseNormal(outputs, name="DenseNormal", eps=self.eps)(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        if self.metrics == "mae":
            metrics = self.mae
        elif self.metrics == "mse":
            metrics = self.mse
        else:
            metrics = None
        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )
        # self.training_var = [np.var(outputs[:, i]) for i in range(outputs.shape[1])]

    def mae(self, y_true, y_pred):
        mu, aleatoric = tf.split(y_pred, 2, axis=-1)
        return tf.keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        mu, aleatoric = tf.split(y_pred, 2, axis=-1)
        return tf.keras.metrics.mean_squared_error(y_true, mu)

    def calc_uncertainties(self, preds, y_scaler=False):
        mu, aleatoric = np.split(preds, 2, axis=-1)
        if len(mu.shape) == 1:
            mu = np.expand_dims(mu)
            aleatoric = np.expand_dims(aleatoric)
        if y_scaler:
            mu = y_scaler.inverse_transform(mu)
        for i in range(aleatoric.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
        return mu, aleatoric
    
    @classmethod
    def load_model(cls, conf):
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        monte_carlo_passes = conf["ensemble"]["monte_carlo_passes"]
        if n_splits > 1 and n_models == 1:
            mode = "data"
        elif n_splits == 1 and n_models > 1:
            mode = "seed"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        else:
            raise ValueError(
                "For the Gaussian model, only one of n_models or n_splits can be > 1 while the other must be 1"
            )
        save_loc = conf["save_loc"]
        # Check if weights file exists
        weights = os.path.join(save_loc, f"{mode}/models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )
        if conf["model"]["verbose"]:
            logger.info(
                f"Loading a parametric DNN with pre-trained weights from path {weights}"
            )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        model_class.training_var = np.loadtxt(
            os.path.join(os.path.join(save_loc, f"{mode}/models", "training_var.txt"))
        )
        if not isinstance(model_class.training_var, list):
            model_class.training_var = [model_class.training_var]

        return model_class
    
    def predict(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        y_out = self.calc_uncertainties(y_out, scaler)
        return y_out

    def predict_monte_carlo(
        self, x_test, y_test, forward_passes, y_scaler=None, batch_size=None
    ):
        """Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        Parameters
        ----------
        data_loader : object
            data loader object from the data loader module
        forward_passes : int
            number of monte-carlo samples/forward passes
        model : object
            keras model
        n_classes : int
            number of classes in the dataset
        y_scaler : sklearn Scaler
            perform inverse scaler on predicted
        """
        n_samples = x_test.shape[0]
        pred_size = y_test.shape[1]
        _batch_size = self.batch_size if batch_size is None else batch_size
        dropout_mu = np.zeros((forward_passes, n_samples, pred_size))
        dropout_aleatoric = np.zeros((forward_passes, n_samples, pred_size))

        for i in range(forward_passes):
            # output = self.model(x_test, training=True)
            output = [
                self.model(x_test[i : i + _batch_size], training=True)
                for i in range(0, x_test.shape[0], _batch_size)
            ]
            mu, aleatoric = self.calc_uncertainties(
                np.concatenate(output, axis=0), y_scaler
            )
            dropout_mu[i] = mu
            dropout_aleatoric[i] = aleatoric

        return dropout_mu, dropout_aleatoric

    def predict_dist_params(self, x, y_scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        preds = self.model.predict(x, batch_size=_batch_size)
        mu, var = np.split(preds, 2, axis=-1)
        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu, 1)
        if y_scaler is not None:
            mu = y_scaler.inverse_transform(mu)

        return mu, var
    
    def predict_ensemble(self, x, weight_locations, batch_size=None, scaler=None):
        num_models = len(weight_locations)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(weight_locations[0])
            mu, var = self.predict(x, batch_size=batch_size, scaler=scaler)
            output_shape = mu.shape[1:]
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_var = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_mu[0] = mu
            ensemble_var[0] = var
        else:
            output_shape = ()  # Default shape if no models
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_var = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(weight_locations[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            mu, var = self.predict(x, scaler=scaler, batch_size=batch_size)
            ensemble_mu[i + 1] = mu
            ensemble_var[i + 1] = var

        return ensemble_mu, ensemble_var


class CategoricalDNN(object):

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

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        output_activation="softmax",
        optimizer="adam",
        loss="categorical_crossentropy",
        loss_weights=None,
        annealing_coeff=None,
        use_noise=False,
        noise_sd=0.0,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.2,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.0,
        l2_weight=0.0,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        decay=0,
        verbose=0,
        classifier=False,
        random_state=1000,
        callbacks=[],
        balanced_classes=0,
        steps_per_epoch=0,
    ):

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
        self.classifier = classifier
        self.y_labels = None
        self.model = None
        self.random_state = random_state
        self.balanced_classes = balanced_classes
        self.steps_per_epoch = steps_per_epoch

    def build_neural_network(self, inputs, outputs):
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

        self.model = tf.keras.models.Sequential()
        self.model.add(
            Dense(
                inputs,
                activation=self.activation,
                kernel_regularizer=self.kernel_reg,
                name="dense_input",
            )
        )

        for h in range(self.hidden_layers):
            self.model.add(
                Dense(
                    self.hidden_neurons,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_reg,
                    name=f"dense_{h:02d}",
                )
            )
            if self.use_dropout:
                self.model.add(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model.add(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model.add(
            Dense(outputs, activation=self.output_activation, name="dense_output")
        )

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon,
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        self.model.build((self.batch_size, inputs))
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss)

    def fit(self, x_train, y_train, validation_data=None):

        inputs = x_train.shape[-1]
        outputs = y_train.shape[-1]

        if self.loss == "dirichlet":
            for callback in self.callbacks:
                if isinstance(callback, ReportEpoch):
                    # Don't use weights within Dirichelt, it is done below using sample weight
                    self.loss = DirichletEvidentialLoss(
                        callback=callback, name=self.loss
                    )
                    break
            else:
                raise OSError(
                    "The ReportEpoch callback needs to be used in order to run the evidential model."
                )
        self.build_neural_network(inputs, outputs)
        if self.balanced_classes:
            train_idx = np.argmax(y_train, 1)
            training_generator, steps_per_epoch = balanced_batch_generator(
                x_train,
                y_train,
                sample_weight=np.array([self.loss_weights[_] for _ in train_idx]),
                sampler=RandomUnderSampler(),
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            history = self.model.fit(
                training_generator,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                shuffle=True,
            )
        elif self.loss_weights is not None:
            sample_weight = np.array([self.loss_weights[np.argmax(_)] for _ in y_train])
            if not self.steps_per_epoch:
                self.steps_per_epoch = sample_weight.shape[0] // self.batch_size
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                sample_weight=sample_weight,
                steps_per_epoch=self.steps_per_epoch,
                # class_weight={k: v for k, v in enumerate(self.loss_weights)},
                shuffle=True,
            )
        else:
            # if not self.steps_per_epoch:
            #    self.steps_per_epoch = sample_weight.shape[0] // self.batch_size
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                # steps_per_epoch=self.steps_per_epoch,
                shuffle=True,
            )
        return history

    @classmethod
    def load_model(cls, conf):
        weights = os.path.join(conf["save_loc"], "models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                "No saved model exists. You must train a model first. Exiting."
            )

        logger.info(
            f"Loading a CategoricalDNN with pre-trained weights from path {weights}"
        )

        input_features = (
            conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
        )
        output_features = conf["ptypes"]
        model_class = cls(**conf["model"])
        model_class.build_neural_network(len(input_features), len(output_features))
        model_class.model.load_weights(weights)
        return model_class

    def save_model(self, model_path):
        tf.keras.models.save_model(self.model, model_path, save_format="h5")
        return

    def predict(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_proba(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_monte_carlo(self, x, mc_forward_passes=10, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self.model(tf.expand_dims(lx, axis=-1), training=True)
                        for lx in np.array_split(x, x.shape[0] // _batch_size)
                    ]
                )
                for _ in range(mc_forward_passes)
            ]
        )
        pred_probs = y_prob.mean(axis=0)
        epistemic = y_prob.var(axis=0)
        aleatoric = np.mean(y_prob * (1.0 - y_prob), axis=0)

        # Calculating entropy across multiple MCD forward passes
        epsilon = sys.float_info.min
        entropy = -np.sum(
            pred_probs * np.log(pred_probs + epsilon), axis=-1
        )  # shape (n_samples,)
        # Calculating mutual information across multiple MCD forward passes
        mutual_info = entropy - np.mean(
            np.sum(-y_prob * np.log(y_prob + epsilon), axis=-1), axis=0
        )  # shape (n_samples,)
        return pred_probs, aleatoric, epistemic, entropy, mutual_info
    
    def predict_ensemble(self, x, weight_locations, batch_size=None):
        num_models = len(weight_locations)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(weight_locations[0])
            first_prediction = self.predict(x, batch_size=batch_size)
            output_shape = first_prediction.shape[1:]
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)
            predictions[0] = first_prediction
        else:
            output_shape = ()  # Default shape if no models
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(weight_locations[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            y_prob = model_instance.predict(x, batch_size=batch_size)
            predictions[i + 1] = y_prob

        return predictions

    def compute_uncertainties(self, y_pred, num_classes=4):
        return calc_prob_uncertainty(y_pred, num_classes=num_classes)


def calc_prob_uncertainty(y_pred, num_classes=4):
    evidence = tf.nn.relu(y_pred)
    alpha = evidence + 1
    S = tf.keras.backend.sum(alpha, axis=1, keepdims=True)
    u = num_classes / S
    prob = alpha / S
    epistemic = prob * (1 - prob) / (S + 1)
    aleatoric = prob - prob**2 - epistemic
    return prob, u, aleatoric, epistemic


def locate_best_model(filepath, metric="val_ave_acc", direction="max"):
    filepath = glob.glob(os.path.join(filepath, "models", "training_log_*.csv"))
    func = min if direction == "min" else max
    scores = defaultdict(list)
    for filename in filepath:
        f = pd.read_csv(filename)
        best_ensemble = int(filename.split("_log_")[1].strip(".csv"))
        scores["best_ensemble"].append(best_ensemble)
        scores["metric"].append(func(f[metric]))

    best_c = scores["metric"].index(func(scores["metric"]))
    return scores["best_ensemble"][best_c]
