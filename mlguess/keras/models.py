import sys
import keras
import keras.ops as ops
import numpy as np
from keras.regularizers import L1, L2, L1L2
from keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from mlguess.keras.layers import DenseNormalGamma, DenseNormal
from mlguess.keras.losses import evidential_cat_loss, evidential_reg_loss, gaussian_nll
from mlguess.keras.callbacks import ReportEpoch
from keras.optimizers import Adam, SGD


@keras.saving.register_keras_serializable()
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

    def __init__(self, hidden_layers=2, hidden_neurons=64, evidential=False, activation="relu",
                 output_activation="softmax", optimizer="adam", loss="categorical_crossentropy", loss_weights=None,
                 annealing_coeff=1.0, use_noise=False, noise_sd=0.0, lr=0.001, use_dropout=False, dropout_alpha=0.2,
                 batch_size=128, epochs=2, kernel_reg=None, l1_weight=0.0, l2_weight=0.0, sgd_momentum=0.9,
                 adam_beta_1=0.9, adam_beta_2=0.999, epsilon=1e-7, decay=0, verbose=0, random_state=1000, n_classes=2,
                 n_inputs=42, callbacks=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.evidential = evidential
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
        self.decay = decay
        self.verbose = verbose
        self.random_state = random_state
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        self.callbacks = callbacks
        self.hyperparameters = ["hidden_layers", "hidden_neurons", "evidential", "activation", "output_activation",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon", "loss",
                                "loss_weights", "annealing_coeff", "lr", "kernel_reg", "l1_weight", "l2_weight",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "callbacks", "decay", "verbose", "random_state", "n_classes", "n_inputs"]
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

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr,
                                      beta_1=self.adam_beta_1,
                                      beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        if self.evidential:
            self.output_activation = "linear"

        self.model_layers = []
        self.model_layers.append(Dense(self.n_inputs,
                                       activation=self.activation,
                                       kernel_regularizer=self.kernel_reg,
                                       name="input_dense"))
        for h in range(self.hidden_layers):
            self.model_layers.append(Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model_layers.append(Dense(self.n_classes,
                                       activation=self.output_activation,
                                       name="dense_output"))

    def call(self, inputs):

        mod = self.model_layers[0](inputs)
        for l in range(1, len(self.model_layers)):
            mod = self.model_layers[l](mod)

        return mod

    def fit(self, x, y, **kwargs):

        if self.evidential:
            e = keras.Variable(1)
            report_epoch_callback = ReportEpoch(e)
            self.loss = evidential_cat_loss(evi_coef=self.annealing_coeff,
                                            epoch_callback=report_epoch_callback)
            self.callbacks = [report_epoch_callback]

        super().compile(loss=self.loss,
                        optimizer=self.optimizer_obj,
                        run_eagerly=False)

        hist = super().fit(x, y, epochs=self.epochs, batch_size=self.batch_size,
                           callbacks=self.callbacks, **kwargs)

        return hist

    def predict(self, x, return_uncertainties=True, **kwargs):

        if (not self.evidential) and return_uncertainties:
            raise NotImplementedError("You can only return uncertainty estimates when 'evidential' is True. Otherwise "
                                      "you can set 'return_uncertainties' to False to return probabilities.")
        output = super().predict(x, **kwargs)
        if return_uncertainties:
            return self.calc_uncertainty(output)
        else:
            return output

    def calc_uncertainty(self, y_pred):
        num_classes = y_pred.shape[-1]
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        S = ops.sum(alpha, axis=1, keepdims=True)
        u = num_classes / S
        prob = alpha / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob ** 2 - epistemic
        return prob, u, aleatoric, epistemic

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self(ops.expand_dims(lx, axis=-1), training=True)
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
            pred_probs * np.log(np.maximum(pred_probs, epsilon)), axis=-1
        )  # shape (n_samples,)
        # Calculating mutual information across multiple MCD forward passes
        mutual_info = entropy - np.mean(
            np.sum(-np.array(y_prob) * np.log(np.maximum(y_prob, epsilon)), axis=-1), axis=0
        )  # shape (n_samples,)
        return pred_probs, aleatoric, epistemic, entropy, mutual_info

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}


class RegressorDNN(keras.models.Model):
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
    def __init__(self, hidden_layers=2, hidden_neurons=64, evidential=False, activation="relu", optimizer="adam",
                 loss_weights=None, use_noise=False, noise_sd=0.01, lr=0.00001, use_dropout=False, dropout_alpha=0.1,
                 batch_size=128, loss="mse", epochs=2, kernel_reg=None, l1_weight=0.01, l2_weight=0.01,
                 sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, epsilon=1e-7, verbose=1, training_var=None,
                 n_inputs=10, n_output_tasks=1, evi_coeff=1.0, uncertainty=False, **kwargs):

        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.evidential = evidential
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss_weights = loss_weights
        self.loss = loss
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
        self.training_var = training_var
        self.epsilon = epsilon
        self.n_output_tasks = n_output_tasks
        self.n_inputs = n_inputs
        self.evi_coeff = evi_coeff
        self.uncertainty = uncertainty
        self.N_EVI_OUTPUT_PARAMS = 4
        self.N_DENSE_NORMAL_OUTPUT_PARAMS = 2
        self.hyperparameters = ["hidden_layers", "evidential", "hidden_neurons", "activation", "training_var",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon",
                                "loss_weights", "lr", "kernel_reg", "l1_weight", "l2_weight", "loss",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "verbose", "n_inputs", "n_output_tasks", "epsilon", "evi_coeff", "uncertainty"]

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

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr,
                                      beta_1=self.adam_beta_1,
                                      beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)

        self.model_layers = []
        self.model_layers.append(Dense(self.n_inputs,
                                       activation=self.activation,
                                       kernel_regularizer=self.kernel_reg,
                                       name="input_dense"))
        for h in range(self.hidden_layers):
            self.model_layers.append(Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        if self.evidential:
            self.model_layers.append(DenseNormalGamma(self.n_output_tasks, name="dense_output"))
        elif self.uncertainty:
            self.model_layers.append(DenseNormal(self.n_output_tasks, name="dense_output"))
        else:
            self.model_layers.append(Dense(self.n_output_tasks, name="dense_output"))
    def call(self, inputs):

        layer_output = self.model_layers[0](inputs)
        for l in range(1, len(self.model_layers)):
            layer_output = self.model_layers[l](layer_output)

        return layer_output

    def fit(self, x, y, **kwargs):

        if self.evidential:
            self.loss = evidential_reg_loss(evi_coef=self.evi_coeff)
        elif self.uncertainty:
            self.loss = gaussian_nll
        super().compile(optimizer=self.optimizer_obj, loss=self.loss)
        hist = super().fit(x, y, epochs=self.epochs, batch_size=self.batch_size, **kwargs)
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
        if type(return_uncertainties) != bool:
            raise ValueError("return_uncertainties must be a boolean")

        if (not self.evidential) and (not self.uncertainty) and return_uncertainties:
            raise NotImplementedError("You can only return uncertainty estimates when 'evidential' or 'uncertainty' is"
                                      " True. Otherwise you can set 'return_uncertainties' to False to return predictions.")

        elif (self.evidential) and return_uncertainties:
            return self.calc_uncertainties(super().predict(x, batch_size=batch_size))

        else:
            return super().predict(x, batch_size=batch_size)

    def calc_uncertainties(self, preds):

        mu, v, alpha, beta = np.split(preds, self.N_EVI_OUTPUT_PARAMS, axis=-1)
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


