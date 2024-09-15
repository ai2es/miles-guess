import torch.nn.functional as F
import torch

# Adapted from https://github.com/dougbrion/pytorch-classification-uncertainty/blob/master/losses.py

def get_device():
    """Get the device for PyTorch operations.

    Returns:
        torch.device: The device to use, either "cuda" if CUDA is available, otherwise "cpu".
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    """Apply the ReLU activation function to the input tensor.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the ReLU activation function.
    """
    return F.relu(y)


def exp_evidence(y):
    """Apply the exponential function to the input tensor with clamping.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the exponential function with clamping.
    """
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    """Apply the Softplus activation function to the input tensor.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the Softplus activation function.
    """
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    """Compute the Kullback-Leibler divergence for a Dirichlet distribution.

    Args:
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        num_classes (int): The number of classes.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The KL divergence between the given Dirichlet distribution and a uniform distribution.
    """
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    """Compute the log-likelihood loss for a Dirichlet distribution.

    Args:
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed log-likelihood loss.
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """Compute the mean squared error loss with KL divergence for Dirichlet distributions.

    Args:
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed MSE loss with KL divergence.
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss.

    Args:
        func (callable): Function to apply to alpha (e.g., log, softplus).
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss.
    """
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    if not isinstance(weights, torch.Tensor):
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    else:
        weights = weights.to(device)
        A = torch.sum(weights * y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss with mean squared error.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with MSE.
    """
    if device is None:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, weights, device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss with the logarithm of evidence.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with logarithmic evidence.
    """
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, weights=None, device=None
):
    """Compute the Evidence Deep Learning (EDL) loss with the digamma function of evidence.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with digamma evidence.
    """
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1

    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss


class EDLLoss(torch.nn.Module):
    def __init__(self, func, num_classes, annealing_step, weights=None):
        super(EDLLoss, self).__init__()
        self.func = func
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.weights = weights

    def forward(self, y, alpha, epoch_num, device=None):
        if device is None:
            device = y.device
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        if self.weights is None:
            A = torch.sum(y * (self.func(S) - self.func(alpha)), dim=1, keepdim=True)
        else:
            weights = self.weights.to(device)
            A = torch.sum(weights * y * (self.func(S) - self.func(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1.0, epoch_num / self.annealing_step)
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device=device)
        return A + kl_div


class EDLDigammaLoss(torch.nn.Module):
    def __init__(self, num_classes, annealing_step, weights=None):
        super(EDLDigammaLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.weights = weights

    def forward(self, output, target, epoch_num, device=None):
        if device is None:
            device = output.device
        evidence = F.relu(output)  # Apply ReLU to output
        alpha = evidence + 1
        edl_loss_fn = EDLLoss(torch.digamma, self.num_classes, self.annealing_step, self.weights)
        loss = torch.mean(edl_loss_fn(target, alpha, epoch_num, device))
        return loss
