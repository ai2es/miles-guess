import sys
import numpy as np
import torch
import torch.nn.functional as F
from mlguess.torch.class_losses import relu_evidence


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def monte_carlo_dropout(data_loader,
                        forward_passes,
                        model,
                        n_classes,
                        n_samples,
                        batch_size=1024,
                        uncertainty=False):
    """Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : Object. Data loader object from the data loader module
    forward_passes : Integer. Number of monte-carlo samples/forward passes
    model : Object. Keras model
    n_classes : Integer. Number of classes in the dataset
    n_samples : Integer. Number of samples in the test set
    batch_size : Integer. Number of samples per gradient update
    uncertainty : Boolean. Whether to calculate uncertainty
    """
    results = {}
    dropout_predictions = np.empty((0, n_samples, n_classes))
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for j, (image, label) in enumerate(data_loader):
            output = model.predict(image, batch_size=batch_size, return_numpy=False)
            if n_classes > 1:
                if uncertainty:
                    evidence = relu_evidence(output)
                    alpha = evidence + 1
                    # u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                    output = alpha / torch.sum(alpha, dim=1, keepdim=True)
                else:
                    output = F.softmax(output, dim=1)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.numpy()))
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    results["mean"] = mean
    results["variance"] = variance

    if n_classes > 1:
        epsilon = sys.float_info.min
        # Calculating entropy across multiple MCD forward passes 
        entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

        # Calculating mutual information across multiple MCD forward passes 
        mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                               axis=-1), axis=0)  # shape (n_samples,)

        results["entropy"] = entropy
        results["mutual_info"] = mutual_info

    return results
