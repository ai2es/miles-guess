"""Torch losses for regression models"""

import numpy as np
import torch
import torch.nn.functional as F

tol = torch.finfo(torch.float32).eps


class EvidentialRegressionLoss:
    """Class for computing Evidential Regression Loss, which includes the Normal Inverse Gamma negative log likelihood
    and a regularization term.

    Args:
        coef (float, optional): Coefficient for the regularization term. Defaults to 1.0.
    """

    def __init__(self, coef=1.0):
        self.coef = coef

    def normal_inverse_gamma_nll(self, y, gamma, v, alpha, beta):
        """Compute the Normal Inverse Gamma Negative Log Likelihood (NLL) for Deep Evidential Regression.

        Args:
            y (torch.Tensor): Target values.
            gamma (torch.Tensor): Mean of the Normal-Inverse Gamma distribution.
            v (torch.Tensor): Degrees of freedom of the distribution.
            alpha (torch.Tensor): Shape parameter of the Normal-Inverse Gamma distribution.
            beta (torch.Tensor): Scale parameter of the Normal-Inverse Gamma distribution.

        Returns:
            torch.Tensor: The computed negative log likelihood.

        Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
        Source: https://github.com/hxu296/torch-evidental-deep-learning
        """
        two_blambda = 2 * beta * (1 + v) + tol
        nll = 0.5 * torch.log(np.pi / (v + tol)) \
              - alpha * torch.log(two_blambda + tol) \
              + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda + tol) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        return nll

    def normal_inverse_gamma_reg(self, y, gamma, v, alpha, beta):
        """Compute the Normal Inverse Gamma Regularizer for Deep Evidential Regression.

        Args:
            y (torch.Tensor): Target values.
            gamma (torch.Tensor): Mean of the Normal-Inverse Gamma distribution.
            v (torch.Tensor): Degrees of freedom of the distribution.
            alpha (torch.Tensor): Shape parameter of the Normal-Inverse Gamma distribution.
            beta (torch.Tensor): Scale parameter of the Normal-Inverse Gamma distribution.

        Returns:
            torch.Tensor: The computed regularization term.
        """
        error = F.l1_loss(y, gamma, reduction="none")
        evi = 2 * v + alpha
        return error * evi

    def __call__(self, gamma, v, alpha, beta, y):
        """Compute the total Evidential Regression Loss which is the sum of the negative log likelihood and the regularization term.

        Args:
            gamma (torch.Tensor): Mean of the Normal-Inverse Gamma distribution.
            v (torch.Tensor): Degrees of freedom of the distribution.
            alpha (torch.Tensor): Shape parameter of the Normal-Inverse Gamma distribution.
            beta (torch.Tensor): Scale parameter of the Normal-Inverse Gamma distribution.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: The total computed loss.
        """
        loss_nll = self.normal_inverse_gamma_nll(y, gamma, v, alpha, beta)
        loss_reg = self.normal_inverse_gamma_reg(y, gamma, v, alpha, beta)
        return loss_nll.mean() + self.coef * loss_reg.mean()


# code below based off https://github.com/deargen/MT-ENet


def modified_mse(gamma, nu, alpha, beta, target, reduction='mean'):
    """Compute the Lipschitz Mean Squared Error (MSE) loss as described in "Improving Evidential Deep Learning via Multitask Learning."

    Args:
        gamma (torch.Tensor): Output of the evidential network.
        nu (torch.Tensor): Output of the evidential network.
        alpha (torch.Tensor): Output of the evidential network.
        beta (torch.Tensor): Output of the evidential network.
        target (torch.Tensor): True labels.
        reduction (str, optional): Specifies the reduction to apply to the output. Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.

    Returns:
        torch.Tensor: The computed modified MSE loss.

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829
    """
    mse = (gamma - target) ** 2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    mod_mse = mse * c

    if reduction == 'mean':
        return mod_mse.mean()
    elif reduction == 'sum':
        return mod_mse.sum()
    else:
        return mod_mse


def get_mse_coef(gamma, nu, alpha, beta, y):
    """Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(nu, alpha, beta)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt() / (delta + tol)).detach()
    return torch.clip(c, min=False, max=1.)


def check_mse_efficiency_alpha(nu, alpha, beta):
    """Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829

    Args:
        nu (torch.Tensor): nu output value of the evidential network
        alpha (torch.Tensor): alpha output value of the evidential network
        beta (torch.Tensor): beta output value of the evidential network

    Return:
        partial f / partial alpha(numpy.array)
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)

    """
    right = (torch.exp((torch.digamma(alpha + 0.5) - torch.digamma(alpha))) - 1) * 2 * beta * (1 + nu) / (nu + 1e-8)
    return right.detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta):
    """Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829

    Args:
        gamma (torch.Tensor): gamma output value of the evidential network
        nu (torch.Tensor): nu output value of the evidential network
        alpha (torch.Tensor): alpha output value of the evidential network
        beta (torch.Tensor): beta output value of the evidential network

    Return:
        partial f / partial nu(torch.Tensor)
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu + 1) / (nu + tol)
    return beta * nu_1 / (alpha + tol)


class EvidentialMarginalLikelihood(torch.nn.modules.loss._Loss):
    """Marginal likelihood error of prior network.
    The target value is not a distribution (mu, std), but a just value.

    Reference: Amini et al. 2020 (https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf)
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(EvidentialMarginalLikelihood, self).__init__(size_average, reduce, reduction)

    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Conduct the forward pass through the loss.

        Args:
            gamma (torch.Tensor): gamma output value of the evidential network
            nu (torch.Tensor): nu output value of the evidential network
            alpha (torch.Tensor): alpha output value of the evidential network
            beta (torch.Tensor): beta output value of the evidential network
            target (torch.Tensor): target value

        Return:
            (Tensor) Negative log marginal likelihood of EvidentialNet

        """
        pi = torch.tensor(np.pi)
        x1 = torch.log(pi / (nu + tol)) * 0.5
        x2 = -alpha * torch.log(2. * beta * (1. + nu) + tol)
        x3 = (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + 2. * beta * (1. + nu) + tol)
        x4 = torch.lgamma(alpha + tol) - torch.lgamma(alpha + 0.5 + tol)
        if self.reduction == 'mean':
            return (x1 + x2 + x3 + x4).mean()
        elif self.reduction == 'sum':
            return (x1 + x2 + x3 + x4).sum()
        else:
            return x1 + x2 + x3 + x4


class EvidenceRegularizer(torch.nn.modules.loss._Loss):
    """Regularization for the regression prior network.
    If self.factor increases, the model output the wider(high confidence interval) predictions.

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/deargen/MT-ENet/tree/468822188f52e517b1ee8e386eea607b2b7d8829
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', coef=0.1):
        super(EvidenceRegularizer, self).__init__(size_average, reduce, reduction)
        self.coef = coef

    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the loss.

        Args:
            gamma (torch.Tensor): gamma output value of the evidential network
            nu (torch.Tensor): nu output value of the evidential network
            alpha (torch.Tensor): alpha output value of the evidential network
            target (torch.Tensor): target value

        Returns:
            (Tensor) prior network regularization
            Loss = |y - gamma|*(2*nu + alpha) * factor

        """
        loss_value = torch.abs(target - gamma) * (2 * nu + alpha) * self.coef
        if self.reduction == 'mean':
            return loss_value.mean()
        elif self.reduction == 'sum':
            return loss_value.sum()
        else:
            return loss_value


class LipschitzMSELoss(torch.nn.Module):
    """Compute the Lipschitz MSE loss, which includes the Evidential Marginal Likelihood, Evidence Regularizer,
    and a modified MSE term.

    Args:
        tol (float, optional): Tolerance value to avoid division by zero. Defaults to 1e-8.
        coef (float, optional): Coefficient for the regularization term. Defaults to 0.1.
        reduction (str, optional): Specifies the method to reduce the loss over the batch. Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.
    """

    def __init__(self, tol=1e-8, coef=0.1, reduction='mean'):
        super(LipschitzMSELoss, self).__init__()
        self.tol = tol
        self.coef = coef
        self.reduction = reduction
        self.evidential_marginal_likelihood = EvidentialMarginalLikelihood(reduction=reduction)
        self.evidence_regularizer = EvidenceRegularizer(coef=coef, reduction=reduction)

    def forward(self, gamma, nu, alpha, beta, target):
        """Compute the total Lipschitz MSE Loss.

        Args:
            gamma (torch.Tensor): Output value of the evidential network for gamma.
            nu (torch.Tensor): Output value of the evidential network for nu.
            alpha (torch.Tensor): Output value of the evidential network for alpha.
            beta (torch.Tensor): Output value of the evidential network for beta.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: The total computed loss.
        """
        loss = self.evidential_marginal_likelihood(gamma, nu, alpha, beta, target)
        loss += self.evidence_regularizer(gamma, nu, alpha, target)
        loss += modified_mse(gamma, nu, alpha, beta, target, reduction=self.reduction)
        return loss
