import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import brier_score_loss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import logging


logger = logging.getLogger(__name__)


def uq_results(df, save_location=None, prefix=None):

    true_labels = df["true_label"].values
    pred_probs = df[[f"pred_conf{k+1}" for k in range(4)]].values

    if "entropy" in df:
        evidential = df["entropy"]
        df_cols = ["aleatoric", "epistemic", "total", "entropy"]
        uq_names = ["Aleatoric", "Epistemic", "Total", "Entropy"]
    else:
        evidential = df["evidential"]
        df_cols = ["aleatoric", "epistemic", "total", "evidential"]
        uq_names = ["Aleatoric", "Epistemic", "Total", "Evidential"]

    df["total"] = df["aleatoric"] + df["epistemic"]
    df["epistemic"] = np.sqrt(df["epistemic"])
    df["aleatoric"] = np.sqrt(df["aleatoric"])
    df["total"] = np.sqrt(df["total"])

    # Compare uncertainties
    if "entropy" in df:
        plot_uncertainties(
            df,
            ["aleatoric", "total"],
            ["epistemic", "ave_entropy"],
            x_labels=["Epistemic", "Entropy"],
            y_labels=["Aleatoric", "Total"],
            num_bins=20,
            save_location=save_location
        )
    else:
        plot_uncertainties(
            df,
            ["aleatoric", "total"],
            ["epistemic", "evidential"],
            x_labels=["Epistemic", "Evidential"],
            y_labels=["Aleatoric", "Total"],
            num_bins=20,
            save_location=save_location
        )

    # Create reliability diagrams for each class
    classifier_attribution(
        true_labels, pred_probs, save_location=save_location, prefix=prefix
    )

    # Create skill-score figures
    classifier_skill_scores(
        true_labels,
        pred_probs,
        df["aleatoric"],
        df["epistemic"],
        df["total"],
        evidential,
        num_bins=10,
        legend_cols=uq_names,
        save_location=save_location,
        prefix=prefix,
    )

    # Create discard fraction figures for each class
    classifier_discard_fraction(
        df,
        save_location=save_location,
        plt_titles=uq_names,
        uncertainty_cols=df_cols,
        prefix=prefix,
    )


def sort_arr(true_labels, pred_probs, confidences, n_bins=10, data_min = False, data_max = False):
    
    # Compute the minimum and maximum values
    if not data_min:
        data_min = np.min(confidences)
    if not data_max:
        data_max = np.max(confidences)

    # Compute the range and standard deviation
    data_range = data_max - data_min
    data_std = np.std(confidences)

    # Use np.geomspace if the range of values is large compared to the standard deviation
    if data_range > 10 * data_std:
        bins = np.geomspace(data_min, data_max, n_bins + 1)

    # Use np.linspace if the range of values is small compared to the standard deviation
    else:
        bins = np.linspace(data_min, data_max, n_bins + 1)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    pred_labels = np.argmax(pred_probs, axis=1)
    results = defaultdict(list)
    for i in range(n_bins):
        condition = np.logical_and(confidences >= bins[i], confidences < bins[i + 1])
        if condition.sum() > 0:
            results["bss"].append(
                brier_multi(true_labels[condition], pred_probs[condition])
            )
            results["eb"].append(1.0 / np.sqrt(condition.sum()))
            results["acc"].append(
                (true_labels[condition] == pred_labels[condition]).mean()
            )
            results["count"].append(condition.sum())
            results["bin"].append(bin_centers[i])

    return results["bin"], results


def brier_multi(targets, probs, num_classes=4, skill_score=True):
    # Create one-hots of the target to deal with multi-class problems
    one_hot = np.zeros((targets.size, num_classes))
    one_hot[np.arange(targets.size), targets] = 1
    # Compute MSE with one-hots and probabilities
    #res = np.mean((probs - one_hot) ** 2, axis=1)
    res = np.mean((probs - one_hot) ** 2)
    if skill_score:
        tot = np.mean(np.sum((one_hot - np.mean(one_hot)) ** 2, axis=1))
        return 1 - res / tot
    else:
        return res


def brier_skill_score_multiclass(true_labels, pred_probs, skill_score=True):
    num_classes = len(set(true_labels))
    bss_sum = 0

    for class_label in range(num_classes):
        # Convert true labels to binary labels
        binary_labels = [1 if label == class_label else 0 for label in true_labels]

        # Convert pred_probs to a numpy array and select the column corresponding to the current class
        class_probs = np.array(pred_probs)[:, class_label]

        # Calculate Brier Score of the predicted probabilities for this class
        bs = brier_score_loss(binary_labels, class_probs)

        # Calculate Brier Score of the reference model for this class
        freq = sum(binary_labels) / len(binary_labels)
        ref_probs = [freq] * len(binary_labels)
        bs_ref = brier_score_loss(binary_labels, ref_probs)

        # Calculate Brier Skill Score for this class and add to the sum
        if skill_score:
            bss = 1 - (bs / bs_ref)
        else:
            bss = bs
        bss_sum += bss

    # Average the Brier Skill Scores over all classes
    bss_avg = bss_sum / num_classes

    return bss_avg


def plot_uncertainties(
    df,
    input_cols,
    output_cols,
    num_bins=20,
    legend_cols=None,
    x_labels=None,
    y_labels=None,
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

    not_nan = np.isfinite(df[input_cols + output_cols])
    df = df[not_nan].copy()
    
    # Loop over each element in output_cols and create a hexbin plot
    for i, (i_col, o_col) in enumerate(zip(input_cols, output_cols)):
        # Calculate the mean prediction for the current column
        y_array = df[i_col].copy()
        x_array = df[o_col].copy()

        # Create the 2D histogram plot
        my_range = [
            [np.nanpercentile(x_array, 5), np.nanpercentile(x_array, 95)],
            [np.nanpercentile(y_array, 5), np.nanpercentile(y_array, 95)],
        ]
        hb = axs[i].hist2d(
            x_array,
            y_array,
            bins=num_bins,
            cmap="inferno",
            range=my_range,
            norm=colors.LogNorm(),
        )

        # Set the axis labels
        # axs[i].set_title(legend_cols[i], fontsize=fontsize)
        axs[i].set_xlabel(x_labels[i], fontsize=fontsize)

        axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
        axs[i].tick_params(axis="both", which="major", labelsize=fontsize)

        # Move the colorbar below the x-axis label
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(hb[3], cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        # Set the tick labels to use scientific notation
        axs[i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

        # Add 1-1 line
        smaller = min(my_range[0][0], my_range[1][0])
        larger = max(my_range[1][0], my_range[1][1])
        axs[i].plot(
            np.linspace(smaller, larger, 10),
            np.linspace(smaller, larger, 10),
            ls="--",
            c="k",
        )

    # make it pretty
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "compare_uncertanties.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_attribution(
    true_labels,
    pred_probs,
    num_bins=10,
    legend_cols=["Rain", "Snow", "Sleet", "Frz Rain"],
    save_location=False,
    prefix=False,
):

    fig, axs = plt.subplots(2, 4, figsize=(10, 5), sharey="row", sharex="col")
    confidences = np.max(pred_probs, axis=1)

    for true_label in sorted(set(list(true_labels))):
        # Top row figures
        c = true_labels == true_label
        bin_centers, results = sort_arr(
            true_labels[c], pred_probs[c], confidences[c], 50
        )
        _ = axs[0][true_label].hist2d(
            bin_centers,
            results["acc"],
            weights=results["count"],
            bins=num_bins,
            range=((0, 1), (0, 1)),
            cmap="inferno",
        )
        axs[0][true_label].plot(
            np.linspace(0, 1, 10), np.linspace(0, 1, 10), "b:", lw=3
        )
        axs[0][true_label].set_title(legend_cols[true_label], fontsize=10)

        # Bottom row
        c = true_labels == true_label
        bin_centers, results = sort_arr(
            true_labels[c], pred_probs[c], confidences[c], num_bins
        )
        axs[1][true_label].errorbar(
            bin_centers, results["acc"], yerr=results["eb"], color="r"
        )

        bin_centers = np.linspace(0, 1, 100)
        ave_true = np.mean(true_labels == true_label)
        ave_true_range = np.array([ave_true for x in bin_centers])

        no_skill = [0.5 * x1 + 0.5 * ave_true for x1 in bin_centers]
        axs[1][true_label].plot(bin_centers, no_skill, "b-")
        axs[1][true_label].plot(
            np.linspace(0, 1, len(ave_true_range)), ave_true_range, ls=":", color="k"
        )
        axs[1][true_label].plot(
            ave_true_range, np.linspace(0, 1, len(ave_true_range)), ls=":", color="k"
        )
        axs[1][true_label].plot(bin_centers, bin_centers, ls=":", color="k")

        fill_cond = np.where(bin_centers < ave_true)[0]
        axs[1][true_label].fill_between(
            np.array(bin_centers)[fill_cond],
            0,
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )
        fill_cond = np.where(bin_centers >= ave_true)[0]
        axs[1][true_label].fill_between(
            np.array(bin_centers)[fill_cond],
            1,
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )
        axs[1][true_label].set_xlim([0, 1.01])
        axs[1][true_label].set_ylim([0, 1.01])
        # axs[1][true_label].set_title(legend_cols[true_label], fontsize = 10)
        axs[1][true_label].set_xlabel("Predicted Probability")

    axs[0][0].set_ylabel("Conditional Observed\n Probability")
    axs[1][0].set_ylabel("Conditional Observed\n Probability")

    if save_location:
        if prefix:
            name = f"class_attr_{prefix}.pdf"
        else:
            name = "class_attr.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_skill_scores(
    true_labels,
    pred_probs,
    aleatoric,
    epistemic,
    total,
    evidential,
    num_bins=10,
    legend_cols=["Aleatoric", "Epistemic", "Total", "Evidential"],
    save_location=False,
    prefix=False,
):

    fig, axs = plt.subplots(1, 4, figsize=(10, 3.5), sharey="row")

    for k, uq in enumerate([aleatoric, epistemic, total, evidential]):
        bin_centers, results = sort_arr(
            true_labels, pred_probs, uq, num_bins * num_bins
        )
        rescaled_bin_centers = (bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
        _ = axs[k].hist2d(
            rescaled_bin_centers,
            results["bss"],
            weights=results["count"],
            bins=(num_bins, num_bins),
            range=((0, 1), (-0.25, 1)),
            cmap="inferno",
        )
        axs[k].plot(np.linspace(0, 1, 10), np.linspace(1, 0, 10), "b:", lw=3)
        axs[k].set_title(legend_cols[k], fontsize=10)
        axs[k].set_xlabel("Coverage", fontsize=10)
        # axs[k].set_xscale("log")

        ax2 = axs[k].twiny()
        ax2.set_xlim(min(uq), max(uq))
        # ax2.set_xticklabels = bin_centers

    axs[0].set_ylabel("Brier Skill Score", fontsize=10)

    plt.tight_layout()
    if save_location:
        if prefix:
            name = f"class_skill_{prefix}.pdf"
        else:
            name = "class_skill.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_discard_fraction(
    df,
    num_bins=10,
    num_classes=4,
    uncertainty_cols=["aleatoric", "epistemic", "total", "evidential"],
    legend_cols=["Rain", "Snow", "Sleet", "Frz Rain"],
    plt_titles=["Aleatoric", "Epistemic", "Total", "Evidential"],
    colors=["#f8d605", "#ce4912", "#042c71", "b", "g"],
    save_location=False,
    prefix=False,
):

    fig, axs = plt.subplots(1, 4, figsize=(10, 3.5), sharey="row")
    db = (1.0 / num_classes) * num_bins
    # df["total"] = np.sqrt(df["aleatoric"] + df["epistemic"])

    for k, uq in enumerate(uncertainty_cols):

        dg = df.copy().sort_values(uq)

        for class_label in range(num_classes):
            dr = dg[dg["true_label"] == class_label].copy()
            results = defaultdict(list)
            percentage = np.linspace(0, 1, dr.shape[0])
            for percent in np.linspace(5, 105, num_bins):
                c = percentage < percent / 100.0
                if c.sum() == 0:
                    continue
                acc = (dr[c]["true_label"] == dr[c]["pred_label"]).mean()
                # bss = dr[c]["bss"].mean()
                bss = brier_multi(
                    dr[c]["true_label"].values,
                    dr[c][[f"pred_conf{k+1}" for k in range(num_classes)]].values,
                )
                results["acc"].append(acc)
                results["bss"].append(bss)
                results["frac"].append(percent)

            axs[k].bar(
                [100 - x + class_label * db for x in results["frac"]],
                results["bss"],
                db,
                color=colors[class_label],
            )

        axs[k].set_xlabel("Fraction removed")
        axs[k].set_title(plt_titles[k])
        axs[k].legend(legend_cols, loc="best")
        # axs[k].set_yscale("log")

    axs[0].set_ylabel("Brier Skill Score")

    plt.tight_layout()

    if save_location:
        if prefix:
            name = f"discard_fraction_{prefix}.pdf"
        else:
            name = "discard_fraction.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )
