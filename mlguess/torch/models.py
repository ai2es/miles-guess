from torch import nn
from torch.nn import init
import torch
import logging
from torch.nn.utils import spectral_norm as SpectralNorm
import warnings
import random
import numpy as np
import os
from mlguess.torch.checkpoint import load_model_state
from mlguess.torch.layers import LinearNormalGamma

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_device():
    """Determine the computing device to use.

    Checks if CUDA is available and returns the appropriate device
    (either "cuda" or "cpu").

    Returns:
        torch.device: The device to be used for computation.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def seed_everything(seed=1234):
    """Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators. Default is 1234.

    Notes:
        This function seeds:
        - Python's random module
        - NumPy's random module
        - PyTorch's CPU and GPU random number generators
        - Sets environment variable to control hash seed
        - Configures PyTorch's cuDNN to be deterministic and benchmark mode.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights using the specified method.

    Args:
        net (nn.Module): The network whose weights are to be initialized.
        init_type (str): The type of initialization method to use. Options are:
                         'normal', 'xavier', 'kaiming', 'orthogonal'. Default is 'normal'.
        init_gain (float): Scaling factor for 'normal', 'xavier', and 'orthogonal' methods. Default is 0.0.
        verbose (bool): If True, logs the initialization method. Default is True.

    Raises:
        NotImplementedError: If an unsupported initialization method is specified.

    Notes:
        - 'normal': Gaussian distribution with mean 0 and specified standard deviation.
        - 'xavier': Xavier initialization.
        - 'kaiming': Kaiming initialization.
        - 'orthogonal': Orthogonal initialization.
    """

    def init_func(m):
        """Initialization function for network layers.

        Args:
            m (nn.Module): The module to be initialized.

        Notes:
            - Applies the specified initialization method to Conv and Linear layers.
            - Initializes BatchNorm2d layers with a normal distribution.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)


class DNN(nn.Module):
    """Initialize the Deep Neural Network (DNN) model.

    Args:
        input_size (int or list of int): Number of input features or a list of sizes for each input.
        output_size (int or list of int): Number of output features or a list of sizes for each output.
        layer_size (list of int): List of sizes for hidden layers. Default is [1000].
        dr (list of float): Dropout rates for each layer. Default is [0.5].
        batch_norm (bool): Whether to use batch normalization. Default is True.
        lng (bool): Whether to use LinearNormalGamma layer at the end. Default is False.
        weight_init (bool): Whether to initialize weights. Default is False.
        num_layers (int): Number of layers to create if layer_size is a single number. Default is None.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 layer_size=[1000],
                 dr=[0.5],
                 batch_norm=True,
                 lng=False,
                 softmax=False,
                 weight_init=False,
                 num_layers=None):

        input_size = len(input_size) if isinstance(input_size, (list, tuple)) else input_size
        output_size = len(output_size) if isinstance(output_size, (list, tuple)) else output_size

        super(DNN, self).__init__()
        self.lng = lng

        if num_layers is not None and isinstance(layer_size, (int, float)):
            layer_size = [layer_size] * num_layers
            dr = [dr] * num_layers if isinstance(dr, (int, float)) else dr

        if len(layer_size) > 0:
            blocks = self.block(input_size, layer_size[0], dr[0], batch_norm)
            if len(layer_size) > 1:
                for i in range(len(layer_size) - 1):
                    blocks += self.block(layer_size[i], layer_size[i + 1], dr[i], batch_norm)
            if lng:
                blocks.append(LinearNormalGamma(layer_size[-1], output_size))
            else:
                blocks.append(nn.Linear(layer_size[-1], output_size))
                if softmax:
                    blocks.append(nn.Softmax(dim=-1))
        else:
            if lng:
                blocks = [LinearNormalGamma(input_size, output_size)]
            else:
                blocks = [nn.Linear(input_size, output_size)]
                if softmax:
                    blocks.append(nn.Softmax(dim=-1))

        self.fcn = nn.Sequential(*blocks)
        if weight_init:
            self.apply(self.init_weights)

    def block(self, input_size, output_size, dr, batch_norm):
        """Create a block of layers for the network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            dr (float): Dropout rate.
            batch_norm (bool): Whether to apply batch normalization.

        Returns:
            list: A list of layers constituting the block.
        """
        block = [SpectralNorm(nn.Linear(input_size, output_size))]
        if batch_norm:
            block.append(nn.BatchNorm1d(output_size))
        block.append(nn.LeakyReLU())
        if dr > 0.0:
            block.append(nn.Dropout(dr))
        return block

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.fcn(x)
        return x

    def load_weights(self, weights_path: str) -> None:
        """Load model weights from a file.

        Args:
            weights_path (str): Path to the weights file (.pt).

        Returns:
            None
        """
        logger.info(f"Loading model weights from {weights_path}")

        try:
            checkpoint = torch.load(
                weights_path,
                map_location=lambda storage, loc: storage
            )
            self.load_state_dict(checkpoint["model_state_dict"])
        except Exception as E:
            logger.info(
                f"Failed to load model weights at {weights_path} due to error {str(E)}"
            )

    def predict(self, input, y_scaler=None, return_uncertainties=True, return_tuple=False):
        """Make predictions with the model.

        Args:
            input (torch.Tensor): Input tensor.
            y_scaler (optional, callable): A scaler to inverse transform predictions. Default is None.
            return_uncertainties (bool): Whether to return uncertainties along with predictions. Default is True.
            return_tuple (bool): Whether to return the output as a tuple. Default is False.

        Returns:
            tuple or torch.Tensor:
                - If `return_tuple` is True, returns a tuple (output).
                - Otherwise, returns concatenated output tensor.
        """
        output = self(input)
        if return_uncertainties:
            mu, aleatoric, epistemic, total = self.predict_uncertainty(output, y_scaler=y_scaler)
            return mu, aleatoric, epistemic, total
        return output

    def predict_uncertainty(self, input, y_scaler=None):
        """Estimate uncertainties of predictions.

        Args:
            input (tuple of torch.Tensor): Tuple containing (mu, v, alpha, beta) tensors.
            y_scaler (optional, callable): A scaler to inverse transform predictions. Default is None.

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): Mean predictions.
                - aleatoric (torch.Tensor): Aleatoric uncertainty.
                - epistemic (torch.Tensor): Epistemic uncertainty.
                - total (torch.Tensor): Total uncertainty (aleatoric + epistemic).
        """
        mu, v, alpha, beta = input
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))

        if len(mu.shape) == 1:
            mu = mu.unsqueeze(1)
            aleatoric = aleatoric.unsqueeze(1)
            epistemic = epistemic.unsqueeze(1)

        if y_scaler:
            mu = mu.detach().cpu().numpy()
            mu = y_scaler.inverse_transform(mu)
            mu = torch.from_numpy(mu).to(aleatoric.device)

            # Torch version of some of the sklearn scalers -- this needs updated later
            # MinMaxScaler inverse transform
            # if y_scaler:
            #     min_val = torch.tensor(y_scaler.data_min_, device=mu.device)
            #     max_val = torch.tensor(y_scaler.data_max_, device=mu.device)
            #     mu = mu * (max_val - min_val) + min_val

            for i in range(mu.shape[-1]):
                aleatoric[:, i] *= self.training_var[i]
                epistemic[:, i] *= self.training_var[i]

        return mu, aleatoric, epistemic, aleatoric + epistemic

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=16):
        """Perform Monte Carlo Dropout predictions.

        Args:
            x (torch.Tensor): Input tensor.
            mc_forward_passes (int): Number of Monte Carlo forward passes. Default is 10.
            batch_size (optional, int): Batch size for processing. Default is None.

        Returns:
            tuple: A tuple containing:
                - pred_probs (numpy.ndarray): Mean predicted probabilities.
                - aleatoric (numpy.ndarray): Aleatoric uncertainty.
                - epistemic (numpy.ndarray): Epistemic uncertainty.
                - entropy (numpy.ndarray): Entropy across multiple forward passes.
                - mutual_info (numpy.ndarray): Mutual information across multiple forward passes.
        """
        with torch.no_grad():
            batches = np.array_split(x, np.ceil(x.shape[0] / batch_size).astype(int))
            y_prob = torch.stack(
                [
                    torch.vstack([self(torch.FloatTensor(lx)) for lx in batches])
                    for _ in range(mc_forward_passes)
                ]
            )
        pred_probs = y_prob.mean(axis=0).numpy()
        epistemic = y_prob.var(axis=0).numpy()
        aleatoric = torch.mean(y_prob * (1.0 - y_prob), axis=0).numpy()
        y_prob = y_prob.numpy()

        # Calculating entropy across multiple MCD forward passes
        epsilon = 1e-10  # A small value to avoid log(0)
        entropy = -np.sum(pred_probs * np.log(np.maximum(pred_probs, epsilon)), axis=-1)
        # Calculating mutual information across multiple MCD forward passes
        mutual_info = entropy - np.mean(
            np.sum(-y_prob * np.log(np.maximum(y_prob, epsilon)), axis=-1), axis=0
        )
        return pred_probs, aleatoric, epistemic, entropy, mutual_info

    @classmethod
    def from_config(cls, conf, device="cpu"):
        """Create a model instance from configuration.

        Args:
            conf (dict): Configuration dictionary with model parameters.
            device (str): Device to load the model onto. Default is "cpu".

        Returns:
            DNN: The initialized model.
        """
        # init the model
        model = cls(**conf["model"])

        # load model weights
        model = load_model_state(conf, model, device)

        return model


class CategoricalDNN(DNN):

    def calc_uncertainty(self, y_pred):
        num_classes = y_pred.shape[-1]
        evidence = torch.relu(y_pred)
        alpha = evidence + 1
        S = torch.sum(alpha, axis=1, keepdims=True)
        u = num_classes / S
        prob = alpha / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob ** 2 - epistemic
        return prob, u, aleatoric, epistemic

    def predict(self, x, return_uncertainties=True, **kwargs):
        self.eval()
        with torch.no_grad():
            output = self(x)
            if return_uncertainties:
                return self.calc_uncertainty(output)
            else:
                return output

    def predict_uncertainty(self, input):
        return self.predict(input, return_uncertainties=True)

    def predict_dropout(self, input):
        raise NotImplementedError


if __name__ == "__main__":
    # Model parameters
    input_size = 10
    output_size = 4
    batch_size = 16

    # Initialize the model
    model = CategoricalDNN(input_size=input_size, output_size=output_size)

    # Example input data (batch size 1, input size 10)
    example_data = torch.rand((batch_size, input_size), dtype=torch.float32)

    # Pass data through the model
    predictions = model(example_data)
    print("Predictions:", predictions)

    prob, u, aleatoric, epistemic = model.predict_uncertainty(example_data)
    print("Predictions:", prob)
    print("Evidential u:", u)
    print("Aleatoric u:", aleatoric)
    print("Epistemic u:", epistemic)
