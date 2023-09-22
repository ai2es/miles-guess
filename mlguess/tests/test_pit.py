from mlguess.pit import pit_histogram, pit_deviation
import numpy as np
from scipy.stats import norm

def test_pit_dev_ensemble():
    n_samples = 1000
    n_members = 20
    uniform_ensemble = np.random.random(size=(n_samples, n_members))
    y_true_ens = uniform_ensemble[:, 0]
    y_true_high = np.ones(n_samples) * 2
    y_true_low = np.ones(n_samples) * -1
    dev_ens = pit_deviation(y_true_ens, uniform_ensemble, pred_type="ensemble")
    dev_high = pit_deviation(y_true_high, uniform_ensemble, pred_type="ensemble")
    dev_low = pit_deviation(y_true_low, uniform_ensemble, pred_type="ensemble")
    print(dev_ens, dev_high, dev_low)
    assert dev_ens >= 0 and dev_ens < 1, f"Dev ens ({dev_ens:0.3f}) should be between 0 and 1"
    assert dev_ens < dev_high and dev_ens < dev_low, f"Dev ens ({dev_ens:0.3f}) too high"
    assert dev_high == dev_low , f"Dev high ({dev_high:0.3f})not equal to dev low ({dev_low:0.3f})"
    return

def test_pit_dev_gaussian(): 
    n_samples = 1000
    g_preds = np.zeros((n_samples, 2))
    g_preds[:, 0] = np.random.random(n_samples) * 4 - 2
    g_preds[:, 1] = np.random.random(n_samples) * 2 + 0.5
    y_true_random = np.zeros((n_samples,))
    y_true_high = np.zeros((n_samples,))
    for i in range(n_samples):
        y_true_random[i] = norm.rvs(loc=g_preds[i, 0], scale=g_preds[i, 1])
        y_true_high[i] = norm.ppf(0.999, loc=g_preds[i, 0], scale=g_preds[i, 1])
    dev_rand = pit_deviation(y_true_random, g_preds, pred_type="gaussian")
    dev_high = pit_deviation(y_true_high, g_preds, pred_type="gaussian")
    print(dev_rand, dev_high)
    assert dev_rand < dev_high, f"Dev rand ({dev_rand:0.3f}) is higher than dev high ({dev_high:0.3f})" 
