import logging
import yaml
import shutil
import os
import glob
import warnings
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ptype.reliability import (
    compute_calibration,
    reliability_diagram,
    reliability_diagrams,
)
from ptype.plotting import (
    plot_confusion_matrix,
    coverage_figures,
)
from mlguess.classifier_uq import uq_results

from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from hagelslag.evaluation.MetricPlotter import roc_curve, performance_diagram

from collections import OrderedDict, defaultdict


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def locate_best_model(filepath, metric="val_ave_acc", direction="max"):
    filepath = glob.glob(os.path.join(filepath, "models", "training_log_*.csv"))
    func = min if direction == "min" else max
    scores = defaultdict(list)
    for filename in filepath:
        f = pd.read_csv(filename)
        best_ensemble = int(filename.split("_log_")[1].strip(".csv"))
        scores["best_ensemble"].append(best_ensemble)
        scores["metric"].append(func(f[metric]))

    best_c = scores["metric"].index(func(scores["metric"]))
    return scores["best_ensemble"][best_c]


def evaluate(conf, reevaluate=False):
    input_features = []
    for features in conf["input_features"]:
        input_features += conf[features]
    output_features = conf["output_features"]
    n_splits = conf["ensemble"]["n_splits"]
    save_loc = conf["save_loc"]
    labels = ["rain", "snow", "sleet", "frz-rain"]
    sym_colors = ["blue", "grey", "red", "purple"]
    symbols = ["s", "o", "v", "^"]

    if n_splits == 1:
        data = {
            name: pd.read_parquet(os.path.join(save_loc, "evaluate", f"{name}.parquet"))
            for name in ["train", "val", "test"]
        }
    else:
        best_split = locate_best_model(save_loc, conf["metric"], conf["direction"])
        # Create symlinks for best model / scalers
        # if not os.path.join(save_loc, "models", f"best.h5"):
        try:
            os.symlink(
                os.path.join(save_loc, "models", f"model_{best_split}.h5"),
                os.path.join(save_loc, "models", "best.h5"),
            )
            os.symlink(
                os.path.join(save_loc, "models", f"training_log_{best_split}.csv"),
                os.path.join(save_loc, "models", "best_training_log.csv"),
            )
        except FileExistsError:
            pass
        try:
            os.symlink(
                os.path.join(save_loc, "scalers", f"input_{best_split}.json"),
                os.path.join(save_loc, "scalers", "input.json"),
            )
            os.symlink(
                os.path.join(save_loc, "scalers", f"output_label_{best_split}.json"),
                os.path.join(save_loc, "scalers", "output_label.json"),
            )
            os.symlink(
                os.path.join(save_loc, "scalers", f"output_onehot_{best_split}.json"),
                os.path.join(save_loc, "scalers", "output_onehot.json"),
            )
        except FileExistsError:
            pass
        # Compute uncertainties with the categorical model
        if conf["model"]["loss"] == "categorical_crossentropy":
            _data = {
                name: pd.read_parquet(
                    os.path.join(save_loc, "evaluate", f"{name}_{best_split}.parquet")
                )[["datetime", "lat", "lon"]]
                for name in ["train", "val", "test"]
            }
            tmpdata = {}
            tmpdata["train"] = pd.concat(
                [
                    pd.read_parquet(
                        os.path.join(
                            save_loc, "evaluate", f"train_{best_split}.parquet"
                        )
                    ),
                    pd.read_parquet(
                        os.path.join(save_loc, "evaluate", f"val_{best_split}.parquet")
                    ),
                ]
            )
            tmpdata["test"] = pd.read_parquet(
                os.path.join(save_loc, "evaluate", f"test_{best_split}.parquet")
            )
            # Loop over the data splits
            for name in ["train", "test"]:
                size = tmpdata[name].shape[0]
                ensemble_p = np.zeros((n_splits, size))
                ensemble_std = np.zeros((n_splits, size))

                if conf["model"]["loss"] == "categorical_crossentropy":
                    ensemble_entropy = np.zeros((n_splits, size))
                    ensemble_mutual = np.zeros((n_splits, size))

                # Loop over ensemble of parametric models
                for split in range(n_splits):
                    if name == "test":
                        dfe = pd.read_parquet(
                            os.path.join(
                                save_loc, "evaluate", f"{name}_{split}.parquet"
                            )
                        )
                    else:
                        dfe = [
                            pd.read_parquet(
                                os.path.join(
                                    save_loc, "evaluate", f"train_{split}.parquet"
                                )
                            )
                        ]
                        dfe.append(
                            pd.read_parquet(
                                os.path.join(
                                    save_loc, "evaluate", f"val_{split}.parquet"
                                )
                            )
                        )
                        dfe = pd.concat(dfe, axis=0)
                    ensemble_p[split] = dfe["pred_conf"]
                    if "epistemic" in dfe:
                        ensemble_std[split] = dfe["epistemic"]
                        if conf["model"]["loss"] == "categorical_crossentropy":
                            ensemble_entropy[split] = dfe["entropy"]
                            ensemble_mutual[split] = dfe["mutual_info"]

                # Compute averages, uncertainties
                tmpdata[name]["ave_conf"] = np.mean(ensemble_p, axis=0)
                tmpdata[name]["aleatoric"] = np.mean(ensemble_std, axis=0)
                if "epistemic" in dfe:
                    tmpdata[name]["epistemic"] = np.var(ensemble_p, axis=0)
                    if conf["model"]["loss"] == "categorical_crossentropy":
                        tmpdata[name]["ave_entropy"] = np.mean(ensemble_entropy, axis=0)
                        tmpdata[name]["ave_mutual_info"] = np.mean(
                            ensemble_mutual, axis=0
                        )

            # Reindex the train / val splits according to indices from the best model
            train_data = _data["train"][["datetime", "lat", "lon"]].merge(
                tmpdata["train"], on=["datetime", "lat", "lon"], how="inner"
            )
            valid_data = _data["val"][["datetime", "lat", "lon"]].merge(
                tmpdata["train"], on=["datetime", "lat", "lon"], how="inner"
            )
            test_data = tmpdata["test"].copy()

            data = {"train": train_data, "val": valid_data, "test": test_data}

            del tmpdata

        else:  # Evidential model contains the uncertainties
            data = {
                name: pd.read_parquet(
                    os.path.join(save_loc, "evaluate", f"{name}_{best_split}.parquet")
                )
                for name in ["train", "val", "test"]
            }

    # Compute categorical metrics
    metrics = defaultdict(list)
    for name in data.keys():
        outs = precision_recall_fscore_support(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            average=None,
            labels=range(len(output_features)),
        )
        metrics["split"].append(name)
        for i, (p, r, f, s) in enumerate(zip(*list(outs))):
            class_name = output_features[i]
            metrics[f"{class_name}_precision"].append(p)
            metrics[f"{class_name}_recall"].append(r)
            metrics[f"{class_name}_f1"].append(f)
            metrics[f"{class_name}_support"].append(s)

    # Confusion matrix
    plot_confusion_matrix(
        data,
        labels,
        axis=1,
        normalize=True,
        save_location=os.path.join(save_loc, "plots", "confusion_matrices_axis1.pdf"),
    )
    plot_confusion_matrix(
        data,
        labels,
        axis=0,
        normalize=True,
        save_location=os.path.join(save_loc, "plots", "confusion_matrices_axis0.pdf"),
    )

    test_days = [day for case in conf["case_studies"].values() for day in case]
    test_days_c = data["test"]["day"].isin(test_days)

    test_data = {
        "test": data["test"][~test_days_c],
        "cases": data["test"][test_days_c],
    }

    plot_confusion_matrix(
        test_data,
        labels,
        axis=1,
        normalize=True,
        save_location=os.path.join(
            save_loc, "plots", "confusion_matrices_test_cases_axis1.pdf"
        ),
    )
    plot_confusion_matrix(
        test_data,
        labels,
        axis=0,
        normalize=True,
        save_location=os.path.join(
            save_loc, "plots", "confusion_matrices_test_cases_axis0.pdf"
        ),
    )

    # Reliability
    metric_keys = [
        "avg_accuracy",
        "avg_confidence",
        "expected_calibration_error",
        "max_calibration_error",
    ]
    for name in data.keys():
        # Calibration stats
        results_calibration = compute_calibration(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
        )
        for key in metric_keys:
            metrics[f"bulk_{key}"].append(results_calibration[key])
        # Bulk
        _ = reliability_diagram(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
            dpi=300,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"bulk_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        # Class by class
        results = OrderedDict()
        for label in range(len(output_features)):
            cond = data[name]["true_label"] == label
            results[output_features[label]] = {
                "true_labels": data[name][cond]["true_label"].values,
                "pred_labels": data[name][cond]["pred_label"].values,
                "confidences": data[name][cond]["pred_conf"].values,
            }
            results_calibration = compute_calibration(
                results[output_features[label]]["true_labels"],
                results[output_features[label]]["pred_labels"],
                results[output_features[label]]["confidences"],
                num_bins=10,
            )
            for key in metric_keys:
                metrics[f"{output_features[label]}_{key}"].append(
                    results_calibration[key]
                )

        _ = reliability_diagrams(
            results,
            num_bins=10,
            draw_bin_importance="alpha",
            num_cols=2,
            dpi=100,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"class_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    # Hagelslag
    for name in data.keys():
        rocs = []
        for i in range(len(output_features)):
            forecasts = data[name][f"pred_conf{i+1}"]
            obs = np.where(data[name]["true_label"] == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts, obs)
            rocs.append(roc)
            metrics[f"{output_features[i]}_auc"].append(roc.auc())
            metrics[f"{output_features[i]}_csi"].append(roc.max_csi())
        roc_curve(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"roc_curve_{name}.pdf"),
        )
        performance_diagram(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"performance_{name}.pdf"),
        )

        # Sorting curves
        coverage_figures(
            data[name],
            output_features,
            colors=sym_colors,
            save_location=os.path.join(save_loc, "plots", f"coverage_{name}.pdf"),
        )

        # UQ figures
        uq_results(
            data[name], save_location=os.path.join(save_loc, "metrics"), prefix=name
        )

    # Save metrics
    pd.DataFrame.from_dict(metrics).to_csv(
        os.path.join(save_loc, "metrics", "performance.csv")
    )


if __name__ == "__main__":

    description = "Usage: python evaluate_mlp.py -c model.yml"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    for newdir in ["plots", "metrics"]:
        os.makedirs(os.path.join(save_loc, newdir), exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    evaluate(conf)
