import numpy as np
from scipy.stats import norm, percentileofscore

# path to test /glade/work/schreck/repos/evidential/main/results/evidential/test.csv

def pit_histogram(y_true, y_pred, pred_type="ensemble", bins=10):
    """ 
    Calculate PIT histogram values for different types of ensemble predictions. Predictions
    can take one of three formats:
    * ensemble: array of n deterministic ensemble predictions with shape (n_samples, n_members).
    * gaussian: arary of 1 gaussian distribution prediction (mean and standard deviation) with shape (n_samples, n_params)
    * gaussian_ensemble: array of multiple gaussian distributions for the same prediction with shape (n_samples, n_params, n_members)

    Args:
        y_true: true values with shape (n_samples,)
        y_pred: predictions in format of gaussian, gaussian_ensemble, or ensemble.
        pred_type: 'ensemble', 'gaussian', or 'gaussian_ensemble'
        bins: number of bins or array of specific bin edges. Bins should cover space between 0              and 1.
    
    Returns:
        pit_hist, pit_bins        
    """
    if pred_type == "gaussian":
        assert len(y_pred.shape) == 2 and y_pred.shape[1] == 2, "Pred shape is incorrect for Gaussian distribution"
        pit_quantiles = probability_integral_transform_gaussian(y_true, y_pred)
    elif pred_type == "gaussian_ensemble":
        assert len(y_pred.shape) == 3 and y_pred.shape[1] == 2, "Pred shape is incorrect for Gaussian ensemble"
        pit_quantiles = pit_gaussian_ensemble(y_true, y_pred)
    else:
        assert len(y_pred.shape) == 2, "Ensemble has incorrect array shape"
        pit_quantiles = probability_integral_transform_ensemble(y_true, y_pred)
    pit_hist, pit_bins = pit_histogram_values(pit_quantiles, bins=bins) 
    return pit_hist, pit_bins


def pit_deviation(y_true, y_pred, pred_type="ensemble", bins=10):
    """Runs pit_histogram then calculates the pit_deviation. See docstring for pit_histogram.
    
    """
    pit_hist, pit_bins = pit_histogram(y_true, y_pred, pred_type=pred_type, bins=bins)
    n_bins = pit_hist.shape[0]
    n_samples = y_pred.shape[0]
    pit_deviation = np.sqrt(1 / n_bins * np.sum((pit_hist / n_samples - 1 / n_bins) ** 2))
    return pit_deviation

def pit_deviation_worst(n_bins):
    """Calculate the worst possible PITD score based on the number of bins. Assumes all the forecasts
    end up in one of the outermost bins.
    """
    return np.sqrt(1 / n_bins * ((n_bins - 1) * (1 / n_bins) ** 2 + (1 - 1 / n_bins) ** 2))


def pit_deviation_skill_score(y_true, y_pred, pred_type="ensemble", bins=10):
    """Calculate PITD score relative to the worst possible PITD for a given number of bins.
    Ranges from 0 to 1.
    """
    pitd_score = pit_deviation(y_true, y_pred, pred_type=pred_type, bins=bins)
    pitd_worst = pit_deviation_worst(bins)
    return 1 - pitd_score / pitd_worst


def probability_integral_transform_ensemble(y_true, y_pred_ens):
    """Calculate the probability integral transform quantiles for an ensemble of predictions

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_ens: predicted ensemble values with shape (n_samples, n_ensemble_members)
    
    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    pit_quantiles = np.zeros(y_true.shape)
    for i in range(y_true.shape[0]):
        pit_quantiles[i] = percentileofscore(y_pred_ens[i], y_true[i]) / 100.0
    return pit_quantiles


def probability_integral_transform_gaussian(y_true, y_pred_gaussian):
    """Calculate the probability integral transform quantiles for a single Gaussian distribution.

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_gaussian: predicted Gaussian parameters (mean, stand. dev.) with shape (n_samples, n_params)
    
    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    pit_quantiles = np.zeros(y_true.shape)
    for i in range(y_true.shape[0]):
        pit_quantiles[i] = norm.cdf(y_true[i], y_pred_gaussian[i, 0], y_pred_gaussian[i, 1])
    return pit_quantiles


def pit_gaussian_ensemble(y_true, y_pred_gauss_ens):
    """Calculate the probability integral transform quantile for an ensemble of Gaussian parametric models

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_gauss_ens: ensemble of gaussian predictions (mean and standard deviation) with shape (n_samples, n_params, n_members)
 
    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    pit_quantiles_members = np.zeros((y_true.shape[0], y_pred_gauss_ens.shape[-1]))
    for m in range(y_pred_gauss_ens.shape[-1]):
        pit_quantiles_members[:, m] = probability_integral_transform_gaussian(y_true, y_pred_gauss_ens[:, :, m])
    pit_quantiles = np.mean(pit_quantiles_members, axis=1)
    return pit_quantiles


def pit_histogram_values(pit_quantiles, bins=10):
    pit_hist, pit_bins = np.histogram(pit_quantiles, bins=bins)
    return pit_hist, pit_bins
