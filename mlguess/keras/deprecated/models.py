import os
import sys
import glob
import keras
import keras.ops as ops
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.regularizers import L1, L2, L1L2
from keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from keras.optimizers import Adam, SGD
from mlguess.keras.layers import DenseNormalGamma, DenseNormal
from mlguess.keras.deprecated.losses import EvidentialRegressionLoss, EvidentialRegressionCoupledLoss, gaussian_nll
from mlguess.keras.deprecated.losses import DirichletEvidentialLoss
from mlguess.keras.callbacks import ReportEpoch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator
from collections import defaultdict
import logging

class BaseRegressor(object):
    """A base class for regression models.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
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
        eps=1e-7
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
        self.training_var = []
        self.metrics = metrics
        self.eps = eps
        self.ensemble_member_files = []
        self.history = None

    def build_neural_network(self, inputs, outputs, last_layer="Dense"):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
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

        if last_layer == "Dense":
            nn_model = Dense(outputs, name="dense_last")(nn_model)
        elif last_layer == "DenseNormal":
            nn_model = DenseNormal(outputs, name="DenseNormal", eps=self.eps)(nn_model)
        elif last_layer == "DenseNormalGamma":
            nn_model = DenseNormalGamma(outputs, name="DenseNormalGamma", eps=self.eps)(
                nn_model
            )
        else:
            raise ValueError("Invalid last_layer type. Use 'Dense', 'DenseNormal', or 'DenseNormalGamma'.")

        self.model = Model(nn_input, nn_model)

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        if self.metrics == "mae":
            metrics = [self.mae]
        elif self.metrics == "mse":
            metrics = [self.mse]
        else:
            metrics = None

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )

    def build_from_sequential(self, model, optimizer="adam", loss="mse", metrics=None):
        """Build the neural network model using a Keras Sequential model.

        Args:
            model (tf.keras.Sequential): Keras Sequential model to use.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for the model.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            metrics (list of str or tf.keras.metrics.Metric): Metrics for the model.
        """
        self.model = model

        if optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr)
        elif optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=loss,
            metrics=metrics
        )

    def fit(
        self,
        x,
        y,
        validation_data=None,
        callbacks=None,
        initial_epoch=0,
        steps_per_epoch=None,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,
        **kwargs,
    ):
        """Fit the regression model.

        Args:
            x: Input data
            y: Target data
            validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch
            callbacks: List of callbacks to apply during training
            initial_epoch: Epoch at which to start training (useful for resuming a previous training run)
            steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
            workers: Number of workers to use for data loading
            use_multiprocessing: If True, use ProcessPoolExecutor to load data, which is faster but can cause issues with certain GPU setups. If False, use a ThreadPoolExecutor.
            **kwargs: Additional arguments to be passed to the `fit` method
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_neural_network first.")
        if self.verbose:
            self.model.summary()
        self.training_var = [np.var(y[:, i]) for i in range(y.shape[-1])]
        self.history = self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            **kwargs,
        )

    def save_model(self):
        """Save the trained model to a file.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, self.model_name)
        keras.models.save_model(
            self.model, model_path, save_format="h5"
        )
        logging.info(f"Saved model to {model_path}")

        # Save the training variances
        np.savetxt(
            os.path.join(self.save_path, self.model_name.replace(".h5", "_training_var.txt")),
            np.array(self.training_var),
        )

    @classmethod
    def load_model(cls, conf):
        """Load a trained model using args from a configuration
        """
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a DNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load ensemble weights
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        elif n_splits > 1 and n_models > 1:
            mode = "multi_ensemble"
        else:
            raise ValueError(
                "Incorrect selection of n_splits or n_models. Both must be at greater than or equal to 1."
            )

        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, mode, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def mae(self, y_true, y_pred):
        """Compute the MAE"""
        num_splits = y_pred.shape[-1]
        if num_splits == 4:
            mu, _, _, _ = ops.split(y_pred, num_splits, axis=-1)
        elif num_splits == 2:
            mu, _ = ops.split(y_pred, num_splits, axis=-1)
        else:
            mu = y_pred  # Assuming num_splits is 1
        return keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        """Compute the MSE"""
        num_splits = y_pred.shape[-1]
        if num_splits == 4:
            mu, _, _, _ = ops.split(y_pred, num_splits, axis=-1)
        elif num_splits == 2:
            mu, _ = ops.split(y_pred, num_splits, axis=-1)
        else:
            mu = y_pred  # Assuming num_splits is 1

        return keras.metrics.mean_squared_error(y_true, mu)

    def predict(self, x, scaler=None, batch_size=None):
        """Predict target values for input data.

        Args:
            x (numpy.ndarray): Input data.
            scaler (optional): Scaler object for preprocessing input data (default: None).
            batch_size (optional): Batch size for prediction (default: None).
            y_scaler (optional): Scaler object for post-processing predicted target values (default: None).

        Returns:
            numpy.ndarray: Predicted target values.
        """
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        if scaler:
            if len(y_out.shape) == 1:
                y_out = np.expand_dims(y_out, 1)
            y_out = scaler.inverse_transform(y_out)
        return y_out

    def predict_ensemble(self, x, batch_size=None, scaler=None, num_outputs=1):
        """Predicts outcomes using an ensemble of trained Keras models.

        Args:
            x (numpy.ndarray): Input data for predictions.
            batch_size (int, optional): Batch size for inference. Default is None.
            scaler (object, optional): Scaler object for preprocessing input data. Default is None.
            num_outputs (int, optional): Number of output predictions. Default is 1.

        Returns:
            numpy.ndarray: Ensemble predictions for the input data.
        """
        num_models = len(self.ensemble_member_files)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(self.ensemble_member_files[0])
            first_model.training_var = np.loadtxt(
                self.ensemble_member_files[0].replace(".h5", "_training_var.txt")
            )
            if not isinstance(first_model.training_var, list):
                first_model.training_var = [first_model.training_var]

            if num_outputs == 1:
                mu = self.predict(x, batch_size=batch_size, scaler=scaler)
            elif num_outputs == 2:
                mu, ale = self.predict_uncertainty(x, batch_size=batch_size, scaler=scaler)
            elif num_outputs == 3:
                mu, ale, epi = self.predict_uncertainty(x, batch_size=batch_size, scaler=scaler)

            output_shape = mu.shape[1:]
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            ensemble_mu[0] = mu
            if num_outputs >= 2:
                ensemble_ale = np.empty((num_models,) + (x.shape[0],) + output_shape)
                ensemble_ale[0] = ale
            if num_outputs == 3:
                ensemble_epi = np.empty((num_models,) + (x.shape[0],) + output_shape)
                ensemble_epi[0] = epi
        else:
            output_shape = ()  # Default shape if no models
            ensemble_mu = np.empty((num_models,) + (x.shape[0],) + output_shape)
            if num_outputs >= 2:
                ensemble_ale = np.empty((num_models,) + (x.shape[0],) + output_shape)
            if num_outputs == 3:
                ensemble_epi = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(self.ensemble_member_files[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            model_instance.training_var = np.loadtxt(
                weight_location.replace(".h5", "_training_var.txt")
            )
            if not isinstance(model_instance.training_var, list):
                model_instance.training_var = [model_instance.training_var]

            if num_outputs == 1:
                mu = self.predict(x, batch_size=batch_size, scaler=scaler)
            elif num_outputs == 2:
                mu, ale = self.predict_uncertainty(x, batch_size=batch_size, scaler=scaler)
            elif num_outputs == 3:
                mu, ale, epi = self.predict_uncertainty(x, batch_size=batch_size, scaler=scaler)

            ensemble_mu[i + 1] = mu
            if num_outputs >= 2:
                ensemble_ale[i + 1] = ale
            if num_outputs == 3:
                ensemble_epi[i + 1] = epi

        if num_outputs == 1:
            return ensemble_mu
        elif num_outputs == 2:
            return ensemble_mu, ensemble_ale

        return ensemble_mu, ensemble_ale, ensemble_epi

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None, num_outputs=1):
        """Perform Monte Carlo dropout predictions for the model.

        Args:
            x_test (numpy.ndarray): Input data for prediction.
            forward_passes (int): Number of Monte Carlo forward passes to perform.
            y_scaler (optional): Scaler object for post-processing predicted target values (default: None).
            batch_size (optional): Batch size for prediction (default: None).
            num_outputs (int): Number of output arrays to return (1, 2, or 3).

        Returns:
            tuple: Tuple of arrays containing predicted target values and specified uncertainties.
        """
        n_samples = x_test.shape[0]
        pred_size = self.model.output_shape[-1]
        _batch_size = self.batch_size if batch_size is None else batch_size

        output_arrs = [np.zeros((forward_passes, n_samples, pred_size)) for _ in range(num_outputs)]

        for i in range(forward_passes):
            output = [self.model(x_test[i:i+_batch_size], training=True)
                      for i in range(0, x_test.shape[0], _batch_size)]
            output = np.concatenate(output, axis=0)

            if scaler:
                output = scaler.inverse_transform(output)

            if num_outputs == 1:
                output_arrs[0][i] = output
            else:
                output = self.calc_uncertainties(output, scaler)
                for j in range(num_outputs):
                    output_arrs[j][i] = output[j]

        if num_outputs == 1:
            return output_arrs[0]

        return tuple(output_arrs)

    def calc_uncertainties(self, output, scaler=None):
        raise NotImplementedError

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        raise NotImplementedError


class RegressorDNN(BaseRegressor):
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
        super().__init__(
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            activation=activation,
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            use_noise=use_noise,
            noise_sd=noise_sd,
            lr=lr,
            use_dropout=use_dropout,
            dropout_alpha=dropout_alpha,
            batch_size=batch_size,
            epochs=epochs,
            kernel_reg=kernel_reg,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            sgd_momentum=sgd_momentum,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            verbose=verbose,
            save_path=save_path,
            model_name=model_name,
            metrics=metrics,
        )


class GaussianRegressorDNN(BaseRegressor):
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

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        loss="",
        optimizer="adam",
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
        eps=1e-7
    ):
        """
        Initialize the EvidentialRegressorDNN.

        Args:
            coupling_coef: Coupling coeffient for loss fix
            evidential_coef: Evidential regularization coefficient.

        """
        super().__init__(  # Call the constructor of the base class
            hidden_layers,
            hidden_neurons,
            activation,
            optimizer,
            loss,
            loss_weights,
            use_noise,
            noise_sd,
            lr,
            use_dropout,
            dropout_alpha,
            batch_size,
            epochs,
            kernel_reg,
            l1_weight,
            l2_weight,
            sgd_momentum,
            adam_beta_1,
            adam_beta_2,
            verbose,
            save_path,
            model_name,
            metrics,
        )
        self.eps = eps
        self.loss = gaussian_nll

    def build_neural_network(self, inputs, outputs, last_layer="DenseNormal"):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
        """
        super().build_neural_network(inputs, outputs, last_layer=last_layer)

    @classmethod
    def load_model(cls, conf):
        # Load ensemble weights
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
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
            logging.info(
                f"Loading a Gaussian DNN with pre-trained weights from path {weights}"
            )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        model_class.training_var = np.loadtxt(
            os.path.join(os.path.join(save_loc, f"{mode}/models", "best_training_var.txt"))
        )
        if not isinstance(model_class.training_var, list):
            model_class.training_var = [model_class.training_var]

        # Load ensemble weights
        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, mode, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def calc_uncertainties(self, preds, y_scaler=False):
        mu, aleatoric = np.split(preds, 2, axis=-1)
        if len(mu.shape) == 1:
            mu = np.expand_dims(mu, axis=0)
            aleatoric = np.expand_dims(aleatoric, axis=0)
        if y_scaler is not None:
            mu = y_scaler.inverse_transform(mu)
        for i in range(aleatoric.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
        return mu, aleatoric

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        y_out = self.calc_uncertainties(y_out, scaler)
        return y_out

    def predict_dist_params(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        preds = self.model.predict(x, batch_size=_batch_size)
        mu, var = np.split(preds, 2, axis=-1)
        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu, 1)
        if scaler is not None:
            mu = scaler.inverse_transform(mu)

        return mu, var

    def predict_ensemble(self, x_test, scaler=None, batch_size=None, num_outputs=2):
        return super().predict_ensemble(x_test, scaler=scaler, batch_size=batch_size, num_outputs=num_outputs)

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None, num_outputs=2):
        return super().predict_monte_carlo(x_test, forward_passes, scaler=scaler,
                                           batch_size=batch_size, num_outputs=num_outputs)


class EvidentialRegressorDNN(BaseRegressor):
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
        eps=1e-7
    ):
        """Initialize the EvidentialRegressorDNN.

        Args:
            coupling_coef: Coupling coeffient for loss fix
            evidential_coef: Evidential regularization coefficient.
        """
        super().__init__(  # Call the constructor of the base class
            hidden_layers,
            hidden_neurons,
            activation,
            optimizer,
            loss,
            loss_weights,
            use_noise,
            noise_sd,
            lr,
            use_dropout,
            dropout_alpha,
            batch_size,
            epochs,
            kernel_reg,
            l1_weight,
            l2_weight,
            sgd_momentum,
            adam_beta_1,
            adam_beta_2,
            verbose,
            save_path,
            model_name,
            metrics,
        )
        self.coupling_coef = coupling_coef
        self.evidential_coef = evidential_coef
        self.eps = eps

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
            raise ValueError("loss needs to be one of 'evidentialReg' or 'evidentialFix'")

        logging.info(f"Using loss: {loss}")

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
        """
        super().build_neural_network(inputs, outputs, last_layer="DenseNormalGamma")

    @classmethod
    def load_model(cls, conf):
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading an evidential DNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        model_class.training_var = np.loadtxt(
            os.path.join(os.path.join(conf["model"]["save_path"], "models", "best_training_var.txt"))
        )

        if not model_class.training_var.shape:
            model_class.training_var = np.array([model_class.training_var])

        # Load ensemble if there is one
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def calc_uncertainties(self, preds, y_scaler=None):
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

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        y_out = self.calc_uncertainties(
            y_out, scaler
        )  # todo calc uncertainty for coupled params
        return y_out

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

    def predict_ensemble(self, x_test, scaler=None, batch_size=None):
        return super().predict_ensemble(x_test, scaler=scaler, batch_size=batch_size, num_outputs=3)

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None):
        return super().predict_monte_carlo(x_test, forward_passes,
                                           scaler=scaler, batch_size=batch_size, num_outputs=3)


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
        random_state=1000,
        callbacks=None,
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
        if callbacks is None:
            self.callbacks = []
        self.callbacks = callbacks
        self.decay = decay
        self.verbose = verbose
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

        self.model = keras.models.Sequential()
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

    def build_from_sequential(self, model, optimizer="adam", loss="mse", metrics=None):
        """
        Build the neural network model using a Keras Sequential model.

        Args:
            model (tf.keras.Sequential): Keras Sequential model to use.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for the model.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            metrics (list of str or tf.keras.metrics.Metric): Metrics for the model.

        """
        self.model = model

        if optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon,
            )
        elif optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=loss,
            loss_weights=self.loss_weights,
        )

    def fit(self, x_train, y_train, validation_data=None):

        inputs = x_train.shape[-1]
        outputs = y_train.shape[-1]

        if self.loss == "evidential":
            this_epoch_num = keras.variable(value=0)
            report_epoch_callback = ReportEpoch(self.annealing_coeff, this_epoch_num)
            self.callbacks.insert(0, report_epoch_callback)
            self.loss = DirichletEvidentialLoss(
                callback=report_epoch_callback, name=self.loss, this_epoch_num=this_epoch_num
            )
            self.output_activation = "linear"
        else:
            self.output_activation = "softmax"

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
        elif self.loss_weights is not None and self.loss != "evidential":
            # Allow weights to be used with ev model but they need loaded into the custom loss first
            if self.loss == "evidential":
                logging.warning("Class weights are not being used with the evidential model. They will be supported in a future version.")

            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                class_weight={k: v for k, v in enumerate(self.loss_weights)},
                shuffle=True,
            )
        else:
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
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

        logging.info(
            f"Loading a CategoricalDNN with pre-trained weights from path {weights}"
        )
        input_features = conf["input_features"]
        output_features = conf["output_features"]

        # flag for our ptype model
        if all([x in conf for x in input_features]):
            input_features = [conf[x] for x in input_features]
            input_features = [item for sublist in input_features for item in sublist]

        model_class = cls(**conf["model"])
        model_class.build_neural_network(len(input_features), len(output_features))
        model_class.model.load_weights(weights)

        # Load the path to ensemble weights
        model_class.ensemble_member_files = []
        if conf["ensemble"]["n_splits"] > 1:
            for j in range(conf["ensemble"]["n_splits"]):
                model_class.ensemble_member_files.append(
                    os.path.join(conf["save_loc"], "models", f"model_{j}.h5")
                )

        return model_class

    def save_model(self, model_path):
        keras.models.save_model(self.model, model_path, save_format="h5")
        return

    def predict(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_proba(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self.model(ops.expand_dims(lx, axis=-1), training=True)
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
            np.sum(-np.array(y_prob) * np.log(y_prob + epsilon), axis=-1), axis=0
        )  # shape (n_samples,)
        return pred_probs, aleatoric, epistemic, entropy, mutual_info

    def predict_ensemble(self, x, batch_size=None):
        num_models = len(self.ensemble_member_files)

        # Initialize output_shape based on the first model's prediction
        if num_models > 0:
            first_model = self.model
            first_model.load_weights(self.ensemble_member_files[0])
            first_prediction = self.predict(x, batch_size=batch_size)
            output_shape = first_prediction.shape[1:]
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)
            predictions[0] = first_prediction
        else:
            output_shape = ()  # Default shape if no models
            predictions = np.empty((num_models,) + (x.shape[0],) + output_shape)

        # Predict for the remaining models
        for i, weight_location in enumerate(self.ensemble_member_files[1:]):
            model_instance = self.model
            model_instance.load_weights(weight_location)
            y_prob = model_instance.predict(x, batch_size=batch_size)
            predictions[i + 1] = y_prob

        return predictions

    def predict_uncertainty(self, x):
        num_classes = self.model.output_shape[-1]
        y_pred = self.predict(x)
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        S = ops.sum(alpha, axis=1, keepdims=True)
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
        best_ensemble = int(filename.split("_log_")[1].replace(".csv", ""))
        scores["best_ensemble"].append(best_ensemble)
        scores["metric"].append(func(f[metric]))

    best_c = scores["metric"].index(func(scores["metric"]))
    return scores["best_ensemble"][best_c]