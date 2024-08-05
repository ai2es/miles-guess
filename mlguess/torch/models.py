from torch import nn
from torch.nn import init
import torch
import logging
from torch.nn.utils import spectral_norm as SpectralNorm
import warnings
import random
import numpy as np
import sys
import os
from mlguess.torch.checkpoint import load_model_state

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights.
    Parameters:
        net: Network. Network to be initialized.
        init_type: String. The name of an initialization method: normal | xavier | kaiming | orthogonal.
        init_gain: Float. Scaling factor for normal, xavier and orthogonal.
        verbose: Boolean. Verbosity mode.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
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
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)


class LinearNormalGamma(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = SpectralNorm(nn.Linear(in_channels, out_channels*4))

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)


class DNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 layer_size=[1000],
                 dr=[0.5],
                 batch_norm=True,
                 lng=False,
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
                blocks.append(SpectralNorm(nn.Linear(layer_size[-1], output_size)))
        else:
            if lng:
                blocks = [LinearNormalGamma(input_size, output_size)]
            else:
                blocks = [SpectralNorm(nn.Linear(input_size, output_size))]

        self.fcn = nn.Sequential(*blocks)
        if weight_init:
            self.apply(self.init_weights)

    def block(self, input_size, output_size, dr, batch_norm):
        block = [SpectralNorm(nn.Linear(input_size, output_size))]
        if batch_norm:
            block.append(nn.BatchNorm1d(output_size))
        block.append(nn.LeakyReLU())
        if dr > 0.0:
            block.append(nn.Dropout(dr))
        return block

    def forward(self, x):
        x = self.fcn(x)
        return x

    def load_weights(self, weights_path: str) -> None:

        """
            Loads model weights given a valid weights path

            weights_path: str
            - File path to the weights file (.pt)
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
        output = self(input)
        if return_uncertainties:
            output = self.predict_uncertainty(output, y_scaler=y_scaler)
        if return_tuple:
            return output
        return torch.cat(output, dim=1)

    def predict_uncertainty(self, input, y_scaler=None):
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

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self(np.expand_dims(lx, axis=-1), training=True)
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

    @classmethod
    def from_config(cls, conf, device="cpu"):
        # init the model
        model = cls(**conf["model"])

        # load model weights
        model = load_model_state(conf, model, device)

        return model
