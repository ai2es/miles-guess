import torch
from torch import nn
from torch.nn.utils import spectral_norm as SpectralNorm


class LinearNormalGamma(nn.Module):
    """A linear layer with a Normal-Gamma distribution parameterization.

    This module applies a linear transformation to the input, followed by
    reshaping and parameter extraction for a Normal-Gamma distribution. The
    parameters are used to estimate the mean (`mu`), variance (`logv`), and
    shape parameters of the Gamma distribution (`logalpha` and `logbeta`).

    Attributes:
        linear (nn.Module): A linear layer with spectral normalization.

    Args:
        in_channels (int): The number of input features.
        out_channels (int): The number of output features.
    """

    def __init__(self, in_channels, out_channels, spectral_norm=True):
        """Initializes the LinearNormalGamma module.

        Args:
            in_channels (int): The number of input features.
            out_channels (int): The number of output features.
        """
        super().__init__()
        if spectral_norm:
            self.linear = SpectralNorm(nn.Linear(in_channels, out_channels * 4))
        else:
            self.linear = nn.Linear(in_channels, out_channels * 4)

    def evidence(self, x):
        """Applies a log transformation to the input with a shift.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        """Forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels).

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): The mean of the distribution.
                - v (torch.Tensor): The variance of the distribution.
                - alpha (torch.Tensor): The shape parameter of the Gamma distribution.
                - beta (torch.Tensor): The rate parameter of the Gamma distribution.
        """
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)
