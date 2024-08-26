import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import os
from mlguess.pit import pit_histogram
from matplotlib.colors import LogNorm
import properscoring as ps 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_results(
    df,
    output_cols,
    mu,
    aleatoric,
    epistemic,
    ensemble_mu=False,
    ensemble_type=False,
    legend_cols=["Friction_velocity", "Sensible_heat", "Latent_heat"],
    fn=None,
):
    # Set up some column names
    mu_cols = [f"{x}_pred" for x in output_cols]
    err_cols = [f"{x}_err" for x in output_cols]
    e_cols = [f"{x}_e" for x in output_cols]
    a_cols = [f"{x}_a" for x in output_cols]
    t_cols = [f"{x}_t" for x in output_cols]

    # Add the predictions to the dataframe and compute absolute error
    df[mu_cols] = mu
    df[a_cols] = np.sqrt(aleatoric)
    df[e_cols] = np.sqrt(epistemic)
    df[t_cols] = np.sqrt(aleatoric + epistemic)
    df[err_cols] = np.abs(mu - df[output_cols])

    # Make 2D histogram of the predicted aleatoric and epistemic uncertainties
    plot_uncertainties(
        np.sqrt(aleatoric),
        np.sqrt(epistemic),
        output_cols,
        legend_cols=legend_cols,
        save_location=fn,
    )
    
    # Make 1D versions for total sigma
    rmse_crps_skill_scores(output_cols, df, mu, aleatoric, epistemic, legend_cols, save_location=fn)

    # Compute attributes figure
    regression_attributes(df, output_cols, legend_cols, save_location=fn)

    # Compute calibration curve and MAE versus sorted epistemic uncertainty
    calibration(df, a_cols, e_cols, err_cols, legend_cols, save_location=fn)

    # spread-skill
    plot_skill_score(
        df[output_cols].values,
        mu,
        np.sqrt(aleatoric),
        np.sqrt(epistemic),
        output_cols,
        legend_cols=legend_cols,
        num_bins=20,
        save_location=fn,
    )

    # discard fraction
    discard_fraction(df, output_cols, legend_cols, save_location=fn)

    # Compute gaussian PIT histogram
    pit_figure_gaussian(
        df,
        output_cols,
        mu,
        aleatoric,
        epistemic,
        legend_cols=legend_cols,
        save_location=fn,
    )

    if ensemble_type:
        # Compute ensemble PIT histogram
        pit_figure_ensemble(
            df,
            output_cols,
            ensemble_mu,
            legend_cols=legend_cols,
            title=ensemble_type,
            save_location=fn,
        )


def compute_coverage(df, col="var", quan="error"):
    df = df.copy()
    df = df.sort_values(col, ascending=True)
    df["dummy"] = 1
    df[f"cu_{quan}"] = df[quan].cumsum() / df["dummy"].cumsum()
    df[f"cu_{col}"] = df[col].cumsum() / df["dummy"].cumsum()
    df[f"{col}_cov"] = 1 - df["dummy"].cumsum() / len(df)
    return df


def calibration_curve(df, col="var", quan="error", bins=10):
    obs = df.sort_values(quan, ascending=True).copy()
    obs[f"{quan}_cov"] = 1 - obs["dummy"].cumsum() / len(obs)
    h, b1, b2 = np.histogram2d(obs[f"{col}_cov"], obs[f"{quan}_cov"], bins=bins)
    cov_var = np.arange(0.025, 1.025, 1.0 / float(bins))
    cov_mae = [np.average(cov_var, weights=hi) for hi in h]
    cov_mae_std = [np.average((cov_mae - cov_var) ** 2, weights=hi) for hi in h]
    cov_var_std = [np.average((cov_mae - cov_var) ** 2, weights=hi) for hi in h.T]
    return cov_var, cov_mae, cov_mae_std, cov_var_std


def calibration(
    dataframe, a_cols, e_cols, mae_cols, legend_cols, bins=10, save_location=False
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
    colors = ["r", "g", "b"]
    lcolors = ["pink", "lightgreen", "lightblue"]

    for a_col, e_col, mae_col, col, lcol in zip(
        a_cols, e_cols, mae_cols, colors, lcolors
    ):
        #  Coverage (sorted uncertainty) versus cumulative metric
        df = compute_coverage(dataframe, col=a_col, quan=mae_col)
        ax1.plot(df[f"{a_col}_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)

        df = compute_coverage(dataframe, col=e_col, quan=mae_col)
        ax2.plot(df[f"{e_col}_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)

        dataframe["tot_uncertainty"] = np.sqrt(
            dataframe[f"{a_col}"] ** 2 + dataframe[f"{e_col}"] ** 2
        )
        df = compute_coverage(dataframe, col="tot_uncertainty", quan=mae_col)
        ax3.plot(df["tot_uncertainty_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)

    ax1.set_xlabel("Confidence percentile (Aleatoric)")
    ax2.set_xlabel("Confidence percentile (Epistemic)")
    ax3.set_xlabel("Confidence percentile (Total)")
    ax1.set_ylabel("MAE")

    ax1.legend(legend_cols)
    ax2.legend(legend_cols)
    ax3.legend(legend_cols)

    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "mae_versus_coverage.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_uncertainties(
    ale,
    epi,
    output_cols,
    num_bins=20,
    legend_cols=None,
    fontsize=10,
    save_location=None,
):

    width = 5 if len(output_cols) == 1 else 10
    height = 3.5 if len(output_cols) == 1 else 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols

    # Loop over each element in output_cols and create a hexbin plot
    for i, col in enumerate(output_cols):
        # Calculate the mean prediction for the current column
        aleatoric = ale[:, i]
        epistemic = epi[:, i]
        
        # Remove any NaNs
        aleatoric = aleatoric[np.isfinite(aleatoric)]
        epistemic = epistemic[np.isfinite(epistemic)]

        # Create the 2D histogram plot
        my_range = [
            [np.percentile(epistemic, 5), np.percentile(epistemic, 95)],
            [np.percentile(aleatoric, 5), np.percentile(aleatoric, 95)],
        ]
        hb = axs[i].hist2d(
            epistemic, aleatoric, bins=num_bins, cmap="inferno", range=my_range
        )

        # Set the axis labels
        axs[i].set_title(legend_cols[i], fontsize=fontsize)
        axs[i].set_xlabel("Epistemic", fontsize=fontsize)
        if i == 0:
            axs[i].set_ylabel("Aleatoric", fontsize=fontsize)
        axs[i].tick_params(axis="both", which="major", labelsize=fontsize)

        # Move the colorbar below the x-axis label
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(hb[3], cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        # Set the tick labels to use scientific notation
        axs[i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

    # make it pretty
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "compare_uncertanties.png"),
            dpi=300,
            bbox_inches="tight",
        )


# def spread_skill(df, output_cols, legend_cols, nbins=20, save_location=None):
#     colors = ["r", "g", "b"]
#     uncertainty_cols = ["e", "a"]
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     lower_bounds = defaultdict(list)
#     upper_bounds = defaultdict(list)
#     for k, col in enumerate(output_cols):
#         for j, u in enumerate(uncertainty_cols):

#             upper = max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             lower = min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])

#             #             upper = 1.01 * max(df[f"{col}_{u}"])
#             #             lower = 0.99 * min(df[f"{col}_{u}"])

#             #             lower = (
#             #                 lower
#             #                 if np.isfinite(lower)
#             #                 else 0.99 * min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             #             )
#             #             upper = (
#             #                 upper
#             #                 if np.isfinite(upper)
#             #                 else 1.01 * max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             #             )

#             if upper > 0 and lower > 0 and (np.log10(upper) - np.log10(lower)) > 2:
#                 bins = np.logspace(np.log10(lower), np.log10(upper), nbins)
#             else:
#                 bins = np.linspace(lower, upper, nbins)

#             bin_range = np.digitize(df[f"{col}_{u}"].values, bins=bins, right=True)
#             bin_means = [
#                 df[f"{col}_{u}"][bin_range == i].mean() for i in range(1, len(bins))
#             ]

#             histogram = defaultdict(list)
#             for bin_no in range(1, max(list(set(bin_range)))):
#                 idx = np.where(bin_range == bin_no)
#                 residuals = df[f"{col}_err"].values[idx] ** 2
#                 mean = np.mean(residuals) ** (1 / 2)
#                 std = np.std(residuals) ** (1 / 2)
#                 histogram["bin"].append(bin_means[bin_no - 1])
#                 histogram["mean"].append(mean)
#                 histogram["std"].append(std)

#             axs[j].errorbar(
#                 histogram["bin"], histogram["mean"], yerr=histogram["std"], c=colors[k]
#             )
#             axs[j].legend(legend_cols)
#             lower_bounds[u].append(bin_means[0])
#             upper_bounds[u].append(bin_means[-2])

#     bins = np.linspace(min(lower_bounds["e"]), max(upper_bounds["e"]), nbins)
#     axs[0].plot(bins, bins, color="k", ls="--")
#     bins = np.linspace(min(lower_bounds["a"]), max(upper_bounds["a"]), nbins)
#     axs[1].plot(bins, bins, color="k", ls="--")
#     axs[0].set_xlabel("Spread (Epistemic uncertainty)")
#     axs[1].set_xlabel("Spread (Aleatoric uncertainty)")
#     axs[0].set_ylabel("Skill score (RMSE)")

#     if len(output_cols) > 1:
#         axs[0].set_xscale("log")
#         axs[0].set_yscale("log")
#         axs[1].set_xscale("log")
#         axs[1].set_yscale("log")

#     plt.tight_layout()

#     if save_location:
#         plt.savefig(
#             os.path.join(save_location, "spread_skill.png"),
#             dpi=300,
#             bbox_inches="tight",
#         )


def compute_skill_score(y_true, y_pred, y_std, num_bins=10):
    """Computes the skill score with RMSE on the y-axis and binned spread on the x-axis.

    Parameters
    ----------
    y_true : array-like
        A 1D array of true values.
    y_pred : array-like
        A 1D array of predicted values.
    y_std : array-like
        A 1D array of standard deviations of predicted values.
    num_bins : int, optional
        The number of bins to use for binning the spread.

    Returns:
    -------
    ss : array-like
        A 2D array of skill scores.
    bins : array-like
        A 1D array of bin edges for the spread.
    """
    # Bin the spread
    spread_min, spread_max = np.percentile(y_std, [5, 95])
    if spread_max - spread_min > 20:
        bins = np.geomspace(spread_min, spread_max, num_bins + 1)
    else:
        bins = np.linspace(spread_min, spread_max, num_bins + 1)
    digitized = np.digitize(y_std, bins)

    # Compute the mean RMSE for each bin
    ss = np.zeros((num_bins,))
    count = np.zeros((num_bins,))
    for i in range(num_bins):
        idx = np.where(digitized == i + 1)[0]
        if len(idx) > 0:
            ss[i] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
            count[i] = len(idx)
    return ss, count, bins


def plot_skill_score(
    y_true,
    y_pred,
    y_ale,
    y_epi,
    output_cols,
    num_bins=50,
    legend_cols=None,
    save_location=False,
):
    """Plots the skill score with RMSE on the y-axis and binned spread on the x-axis.

    Parameters
    ----------
    y_true : array-like
        A 1D array of true values.
    y_pred : array-like
        A 1D array of predicted values.
    y_std : array-like
        A 1D array of standard deviations of predicted values.
    num_bins : int, optional
        The number of bins to use for binning the spread.
    """
    num_outputs = len(output_cols)
    if num_outputs == 1:
        y_true = np.expand_dims(y_true, 1)
        y_pred = np.expand_dims(y_pred, 1)
        y_ale = np.expand_dims(y_ale, 1)
        y_epi = np.expand_dims(y_epi, 1)

    width = 10  # 5 if num_outputs == 1 else 10
    height = 3.5 if num_outputs == 1 else 7
    fig, axs = plt.subplots(num_outputs, 3, figsize=(width, height))
    if num_outputs == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols

    unc_lab = ["Aleatoric", "Epistemic", "Total"]

    for j in range(num_outputs):

        y_tot = np.sqrt(y_ale**2 + y_epi**2)

        for i, std in enumerate([y_ale, y_epi, y_tot]):

            # Compute the skill score
            not_nan = np.isfinite(y_pred[:, j]) & np.isfinite(std[:, j])
            
            ss, counts, bins = compute_skill_score(
                y_true[:, j][not_nan], y_pred[:, j][not_nan], std[:, j][not_nan], num_bins
            )

            # Compute bin centers
            x_centers = (bins[:-1] + bins[1:]) / 2
            y_centers = ss

            # Calculate range based on percentile of counts
            my_range = [
                [
                    min(np.percentile(x_centers, 5), np.percentile(y_centers, 5)),
                    np.percentile(x_centers, 95),
                ],
                [
                    min(np.percentile(x_centers, 5), np.percentile(y_centers, 5)),
                    np.percentile(y_centers, 95),
                ],
            ]

            _ = axs[j][i].hist2d(
                x_centers,
                y_centers,
                weights=counts / sum(counts),
                bins=num_bins,
                cmap="inferno",
                range=my_range,
            )

            # Add 1-1 line
            minx = min(min(x_centers), min(y_centers))
            maxx = max(max(x_centers), max(y_centers))
            ranger = np.linspace(minx, maxx, 10)
            axs[j][i].plot(ranger, ranger, c="#FFFFFF", ls="--", lw=3, zorder=10)
            axs[j][i].set_xlim([my_range[0][0], my_range[0][1]])
            axs[j][i].set_ylim([my_range[1][0], my_range[1][1]])

            if i == 0:
                axs[j][i].set_ylabel(f"{legend_cols[j]}\nSkill (RMSE)")
            if j == num_outputs - 1:
                axs[j][i].set_xlabel(f"Spread ({unc_lab[i]})")
            axs[j][i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "spread_skill.png"),
            dpi=300,
            bbox_inches="tight",
        )


def discard_fraction(df, output_cols, legend_cols, save_location=False, fontsize=10):
    width = 7 if len(output_cols) == 1 else 10
    height = 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]
    colors = ["#f8d605", "#ce4912", "#042c71"]

    for col in output_cols:
        df = compute_coverage(df, col=f"{col}_e", quan=f"{col}_err")
        df = compute_coverage(df, col=f"{col}_a", quan=f"{col}_err")
        # df[f"{col}_t"] = np.sqrt(df[f"{col}_a"] + df[f"{col}_e"])
        df = compute_coverage(df, col=f"{col}_t", quan=f"{col}_err")
    for k, col in enumerate(output_cols):
        results = defaultdict(list)
        for percent in range(5, 105, 5):
            c = df[f"{col}_e_cov"] >= percent / 100.0
            results["rmse_e"].append(np.square(df[c][f"{col}_err"]).mean() ** (1 / 2))
            c = df[f"{col}_a_cov"] >= percent / 100.0
            results["rmse_a"].append(np.square(df[c][f"{col}_err"]).mean() ** (1 / 2))
            c = df[f"{col}_t_cov"] >= percent / 100.0
            results["rmse_t"].append(np.square(df[c][f"{col}_err"]).mean() ** (1 / 2))
            results["frac"].append(percent)

        db = (1.0 / 3.0) * 5
        axs[k].bar(results["frac"], results["rmse_e"], db, color=colors[0])
        axs[k].bar(
            [x + db for x in results["frac"]], results["rmse_a"], db, color=colors[1]
        )
        axs[k].bar(
            [x + 2 * db for x in results["frac"]],
            results["rmse_t"],
            db,
            color=colors[2],
        )
        axs[k].set_xlabel("Fraction removed")
        axs[k].set_title(legend_cols[k], fontsize=fontsize)
        axs[k].legend(["Epistemic", "Aleatoric", "Total"], loc="best")

    axs[0].set_ylabel("RMSE")
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "discard_fraction.png"),
            dpi=300,
            bbox_inches="tight",
        )


def regression_attributes(df, output_cols, legend_cols, nbins=11, save_location=False, fontsize=10):
    width = 7 if len(output_cols) == 1 else 10
    height = 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))

    if len(output_cols) == 1:
        axs = [axs]

    for k, col in enumerate(output_cols):
        upper = 1.01 * max(df[f"{col}_pred"])
        lower = 0.99 * min(df[f"{col}_pred"])
        bins = np.linspace(lower, upper, nbins)
        bin_range = np.digitize(df[f"{col}_pred"].values, bins=bins)
        bin_means = [
            df[f"{col}_pred"][bin_range == i].mean() for i in range(1, len(bins))
        ]
        histogram = defaultdict(list)
        for bin_no in range(1, max(list(set(bin_range)))):
            idx = np.where(bin_range == bin_no)
            residuals = df[f"{col}"].values[idx]
            mean = np.mean(residuals)
            std = np.std(residuals)
            if np.isfinite(mean):
                histogram["bin"].append(bin_means[bin_no - 1])
                histogram["mean"].append(mean)
                histogram["std"].append(std)
        axs[k].errorbar(
            histogram["bin"], histogram["mean"], yerr=histogram["std"], c="r", zorder=2
        )
        # axs[k].plot(histogram["bin"], histogram["mean"], c = "r")
        axs[k].plot(histogram["bin"], histogram["bin"], "k--")

        # curve for the no-skill line
        ave_true = np.mean(histogram["mean"])
        ave_true_range = np.array([ave_true for x in histogram["mean"]])
        no_skill = [0.5 * x1 + 0.5 * ave_true for x1 in histogram["bin"]]
        axs[k].plot(histogram["bin"], no_skill, "b-")
        axs[k].plot(histogram["bin"], ave_true_range, ls=":", color="lightgrey")
        axs[k].plot(ave_true_range, histogram["mean"], ls=":", color="lightgrey")

        full_range = np.linspace(min(histogram["bin"]), max(histogram["bin"]), 200)
        no_skill = [0.5 * x1 + 0.5 * ave_true for x1 in full_range]
        fill_cond = np.where(full_range < ave_true)[0]
        axs[k].fill_between(
            np.array(full_range)[fill_cond],
            min(
                min(np.array(histogram["mean"]) - np.array(histogram["std"])),
                min(np.array(full_range)),
            ),
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )
        fill_cond = np.where(full_range > ave_true)[0]
        axs[k].fill_between(
            np.array(full_range)[fill_cond],
            max(
                max(np.array(histogram["mean"]) + np.array(histogram["std"])),
                max(np.array(full_range)),
            ),
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )

        axs[k].set_title(f"{legend_cols[k]}", fontsize=fontsize)
        axs[k].set_ylabel("Conditional mean observation")
        axs[k].set_xlabel("Prediction")

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "regression_attributes.png"),
            dpi=300,
            bbox_inches="tight",
        )


def pit_figure_gaussian(
    df,
    output_cols,
    mu,
    aleatoric,
    epistemic,
    titles=["Aleatoric", "Epistemic", "Total"],
    legend_cols=["Friction velocity", "Sensible heat", "Latent heat"],
    save_location=None,
):

    # Create the figure and subplot
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), sharey="col")

    # add up uncertainties
    total = aleatoric + epistemic

    for j, uq in enumerate([aleatoric, epistemic, total]):

        # Loop over the output columns and plot the histograms
        for i, col in enumerate(output_cols):

            bin_counts, bin_edges = pit_histogram(
                df[col].values,
                np.stack([mu[:, i], np.sqrt(uq[:, i])], -1),
                pred_type="gaussian",
                bins=np.linspace(0, 1, 10),
            )
            bin_width = bin_edges[1] - bin_edges[0]

            # Normalize the bin heights
            bin_heights = bin_counts / bin_width
            bin_heights /= sum(bin_heights)

            # Plot the histogram
            axs[j].bar(
                bin_edges[:-1] + i * (bin_width / len(output_cols)),
                bin_heights,
                width=bin_width / len(output_cols),
                align="edge",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.7,
                label="{}".format(col),
            )

        # Add axis labels and title
        axs[j].set_xlabel("PIT Quantiles", fontsize=10)
        # axs[j].set_ylabel('Normalized Bin Height', fontsize=14)

        # Add a grid
        axs[j].grid(axis="y", linestyle="--", alpha=0.7)

        # Increase the font size of the tick labels
        axs[j].tick_params(axis="both", which="major", labelsize=10)

        # Add a legend
        axs[j].legend(legend_cols, fontsize=8, loc="best")

        # Set the titles
        axs[j].set_title(titles[j], fontsize=10)

    plt.tight_layout()

    # Save
    if save_location:
        plt.savefig(
            os.path.join(save_location, "pit_histogram_gaussian.png"),
            dpi=300,
            bbox_inches="tight",
        )


def pit_figure_ensemble(
    df,
    output_cols,
    mu,
    legend_cols=["Friction velocity", "Sensible heat", "Latent heat"],
    title="Ensemble",
    save_location=None,
):

    # Create the figure and subplot
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5), sharey="col")

    mu = np.transpose(mu, (2, 1, 0))

    # Loop over the output columns and plot the histograms
    for i, col in enumerate(output_cols):

        bin_counts, bin_edges = pit_histogram(
            df[col].values, mu[i], pred_type="ensemble", bins=np.linspace(0, 1, 10)
        )
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize the bin heights
        bin_heights = bin_counts / bin_width
        bin_heights /= sum(bin_heights)

        # Plot the histogram
        axs.bar(
            bin_edges[:-1] + i * (bin_width / len(output_cols)),
            bin_heights,
            width=bin_width / len(output_cols),
            align="edge",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.7,
            label="{}".format(col),
        )

    # Add axis labels and title
    axs.set_xlabel("PIT Quantiles", fontsize=10)

    # Add a grid
    axs.grid(axis="y", linestyle="--", alpha=0.7)

    # Increase the font size of the tick labels
    axs.tick_params(axis="both", which="major", labelsize=10)

    # Add a legend
    axs.legend(legend_cols, fontsize=8, loc="best")

    # Set the titles
    axs.set_title(title, fontsize=10)

    plt.tight_layout()

    # Save
    if save_location:
        plt.savefig(
            os.path.join(save_location, "pit_histogram_ensemble.png"),
            dpi=300,
            bbox_inches="tight",
        )

        
def calculate_skill_score(y_true, y_pred, sigma, num_bins=10, log = False, filter_top_percentile = 5):
    # Create a DataFrame from the provided data
    try:
        df = pd.DataFrame({'y_true': y_true[:,0], 'y_pred': y_pred[:,0], 'sigma': sigma[:,0]})
    except:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sigma': sigma})
        
    # Dont use NaNs
    df=df[np.isfinite(df['sigma'])].copy()
    
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
        
        if len(bin_group["y_true"]) ==0 or len(bin_group["y_pred"]) ==0:
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


def rmse_crps_skill_scores(output_cols, df, mu, aleatoric, epistemic, titles, save_location=None):
    # Create a grid of subplots with the number of rows determined by the length of output_cols
    num_cols = len(output_cols)
    num_rows = 2  # You can adjust the number of rows as needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 7))

    # Loop over the length of output_cols
    for col in range(num_cols):
        result = calculate_skill_score(
            df[output_cols[col]].values,
            mu[:, col],
            np.sqrt(aleatoric[:, col] + epistemic[:, col]),
            num_bins=100,
            log=True
        )

        # Calculate the y-axis limits based on data range
        x_limit = (0.0, 1.05 * result['bin'].max())
        y_limit_rmse = (0.95 * result['rmse'].min(), 1.05 * result['rmse'].max())
        y_limit_crps = (0.95 * result['crps'].min(), 1.05 * result['crps'].max())

        # Check if there is only one column, if so, axes will not be a list
        if num_cols == 1:
            ax_rmse = axes[0]
            ax_crps = axes[1]
        else:
            ax_rmse = axes[0, col]
            ax_crps = axes[1, col]

        # Plot RMSE on the left subplot
        sc_rmse = ax_rmse.scatter(result['bin'], result['rmse'], c=result['count'], cmap='viridis', alpha=0.5, norm=LogNorm())
        if col == 0:
            ax_rmse.set_ylabel('RMSE')
        ax_rmse.plot(result['bin'], result['bin'], c="k", ls="--")
        ax_rmse.set_title(titles[col], fontsize=10)

        # Set y-axis limits for the RMSE subplot
        ax_rmse.set_ylim(y_limit_rmse)

        # Plot CRPS on the right subplot
        sc_crps = ax_crps.scatter(result['bin'], result['crps'], c=result['count'], cmap='viridis', alpha=0.5, norm=LogNorm())
        ax_crps.set_xlabel(r'$\sigma_{Total}$')
        ax_crps.set_ylabel('CRPS')

        # Set y-axis limits for the CRPS subplot
        ax_crps.set_ylim(y_limit_crps)

        # Add colorbar to the right subplot
        if col == 0:
            cbar_ax = fig.add_axes([1.01, 0.09, 0.015, 0.88])
            cbar = fig.colorbar(sc_crps, cax=cbar_ax)
            cbar.set_label('Count')

        # Plot the 1-1 line in the CRPS subplot
        ax_crps.plot(result['bin'], result['bin'], c="k", ls="--")

    # Ensure proper spacing between subplots
    plt.tight_layout()
    
    # Save
    if save_location:
        plt.savefig(
            os.path.join(save_location, "rmse_crps_skill.png"),
            dpi=300,
            bbox_inches="tight",
        )