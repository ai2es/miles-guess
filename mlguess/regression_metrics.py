from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import properscoring as ps
from mlguess.pit import pit_deviation_skill_score, pit_histogram
import traceback
# from mlguess.regression_uq import calculate_skill_score


def regression_metrics(y_true, y_pred, total=None, split="val"):
    """Compute common regression metrics for continuous data.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    dict: A dictionary containing common regression metrics.
    """
    metrics = {}

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    metrics[f'{split}_mse'] = mse

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    metrics[f'{split}_rmse'] = rmse

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    metrics[f'{split}_mae'] = mae

    # Calculate R-squared (R2) score
    r2 = r2_score(y_true, y_pred)
    metrics[f'{split}_r2'] = r2

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    metrics[f'{split}_mape'] = mape

    if total is not None:

        # Add PIT skill-score
        pitd = []
        for i, col in enumerate(range(y_true.shape[1])):
            try:
                # Ensure y_true, y_pred, and total are NumPy arrays
                y_true_np = np.asarray(y_true[:, i])
                y_pred_np = np.asarray(y_pred[:, i])
                total_np = np.asarray(total[:, i])
        
                # Filter out infinity values
                finite_mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np) & np.isfinite(total_np)
                y_true_finite = y_true_np[finite_mask]
                y_pred_finite = y_pred_np[finite_mask]
                total_finite = total_np[finite_mask]
        
                # Calculate PIT histogram
                bin_counts, bin_edges = pit_histogram(
                    y_true_finite,
                    np.stack([y_pred_finite, np.sqrt(total_finite)], -1),
                    pred_type="gaussian",
                    bins=np.linspace(0, 1, 10),
                )
                bin_width = bin_edges[1] - bin_edges[0]
        
                # Normalize the bin heights
                bin_heights = bin_counts / bin_width
                bin_heights /= sum(bin_heights)
                y_pred_stack = np.stack([y_pred_finite, np.sqrt(total_finite)], -1)
                pit_score = pit_deviation_skill_score(y_true_finite, y_pred_stack, pred_type="gaussian", bins=10)
        
            except ValueError:
                pit_score = -1
            pitd.append(pit_score)
            metrics[f"{split}_pitd_{col}"] = pit_score

        metrics[f"{split}_pitd"] = np.mean(pitd)
        metrics[f"{split}_crps"] = ps.crps_gaussian(y_true, mu=y_pred, sig=total).mean()

        # result = calculate_skill_score(
        #     y_true,
        #     y_pred,
        #     total,
        #     num_bins=50,
        #     log=True,
        #     filter_top_percentile = 5
        # )

        # metrics[f"{split}_crps_ss"] = r2_score(result['bin'], result['crps'], sample_weight=result["count"])
        # metrics[f"{split}_rmse_ss"] = r2_score(result['bin'], result['rmse'], sample_weight=result["count"])

        try:
            rmse_ss = rmse_crps_skill_scores(y_true, y_pred, total, filter_top_percentile=0)
        except:
            print(traceback.print_exc())
            rmse_ss = {
                "r2_rmse": -100000,
                "r2_crps": -100000
            }

        metrics[f"{split}_r2_rmse_sigma"] = rmse_ss["r2_rmse"]
        metrics[f"{split}_r2_crps_sigma"] = rmse_ss["r2_crps"]

    return metrics


def rmse_crps_skill_scores(y, mu, total, filter_top_percentile=0):
    # Create a grid of subplots with the number of rows determined by the length of output_cols
    r2_rmse_dict = {}
    r2_crps_dict = {}

    # Loop over the length of output_cols
    num_cols = y.shape[-1]
    for col in range(y.shape[-1]):
        result = calculate_skill_score(
            y[:, col],
            mu[:, col],
            np.sqrt(total)[:, col],
            num_bins=100,
            log=True,
            filter_top_percentile=filter_top_percentile
        )

        r2_rmse = r2_score(result['bin'], result['rmse'])
        r2_crps = r2_score(result['bin'], result['crps'])
        r2_rmse_dict[col] = r2_rmse
        r2_crps_dict[col] = r2_crps

        # if np.isnan(r2_rmse):
        #     r2_rmse = -10

        # # Check if r2_crps is NaN and replace it with -10
        # if np.isnan(r2_crps):
        #     r2_crps = -10

    # Calculate the average of r2_rmse and r2_crps
    avg_r2_rmse = sum(r2_rmse_dict.values()) / num_cols
    avg_r2_crps = sum(r2_crps_dict.values()) / num_cols

    # Create and return a dictionary with average r2_rmse and r2_crps
    avg_scores_dict = {'r2_rmse': avg_r2_rmse, 'r2_crps': avg_r2_crps}
    return avg_scores_dict


def calculate_skill_score(y_true, y_pred, sigma, num_bins=10, log=False, filter_top_percentile=0):
    # Create a DataFrame from the provided data
    try:
        df = pd.DataFrame({'y_true': y_true[:, 0], 'y_pred': y_pred[:, 0], 'sigma': sigma[:, 0]})
    except Exception:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sigma': sigma})

    # Dont use NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Add a small constant to sigma to avoid log(0) issues
    epsilon = 1e-10
    df['sigma'] = df['sigma'] + epsilon

    # Create bins based on the 'sigma' column
    if log:
        df['bin'] = pd.cut(np.log(df['sigma']), bins=num_bins)
    else:
        df['bin'] = pd.cut(df['sigma'], bins=num_bins)

    # Calculate the threshold for the top N% based on 'sigma'
    threshold = np.percentile(df['sigma'], 100 - filter_top_percentile)

    # Filter the DataFrame to keep only data points below the threshold
    df = df[df['sigma'] <= threshold]

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame(columns=['bin', 'rmse', 'crps', 'count'])

    # Iterate over each bin
    for bin_name, bin_group in df.groupby('bin'):

        if len(bin_group["y_true"]) == 0 or len(bin_group["y_pred"]) == 0:
            continue

        # Calculate RMSE for the points within the bin
        rmse = np.sqrt(mean_squared_error(bin_group['y_true'], bin_group['y_pred']))

        # Calculate R2 score for the points within the bin
        crps = ps.crps_gaussian(bin_group['y_true'], mu=bin_group['y_pred'], sig=bin_group['sigma']).mean()

        # Get the left bound of the bin
        bin_left = bin_name.left
        if log:
            bin_left = np.exp(bin_left)

        # Get the count of data points in the bin
        count = len(bin_group)

        # Append the results to the result DataFrame
        result_df = result_df._append({'bin': bin_left, 'rmse': rmse, 'crps': crps, 'count': count}, ignore_index=True)

    return result_df
