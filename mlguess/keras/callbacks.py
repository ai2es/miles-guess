from keras import backend as K
from keras.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
)
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from typing import List, Dict
import logging
from functools import partial
import math
import os
import keras
import numpy as np
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

logger = logging.getLogger(__name__)


def get_callbacks(config: Dict[str, str], path_extend=False) -> List[Callback]:
    callbacks = []

    if "callbacks" in config:
        if path_extend:
            save_data = os.path.join(config["save_loc"], path_extend)
        else:
            save_data = config["save_loc"]
        config = config["callbacks"]
    else:
        return []

    if "ModelCheckpoint" in config:
        config["ModelCheckpoint"]["filepath"] = os.path.join(
            save_data, config["ModelCheckpoint"]["filepath"]
        )
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer")

    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")

    # LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer

    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")

    if "CSVLogger" in config:
        config["CSVLogger"]["filename"] = os.path.join(
            save_data, config["CSVLogger"]["filename"]
        )
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")

    if "LearningRateScheduler" in config:
        drop = config["LearningRateScheduler"]["drop"]
        epochs_drop = config["LearningRateScheduler"]["epochs_drop"]
        f = partial(step_decay, drop=drop, epochs_drop=epochs_drop)
        callbacks.append(LearningRateScheduler(f))
        callbacks.append(LearningRateTracker())

    return callbacks


def step_decay(epoch, drop=0.2, epochs_drop=5.0, init_lr=0.001):
    lrate = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)

@keras.saving.register_keras_serializable()
class ReportEpoch(keras.callbacks.Callback):
    def __init__(self, epoch_var):
        self.epoch_var = epoch_var

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_var += 1

    def get_config(self):

        return {}


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, x, y, name="val", n_bins=10, use_uncertainty=False, **kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.name = name
        self.use_uncertainty = use_uncertainty

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def on_epoch_end(self, epoch, logs={}):
        if self.use_uncertainty:
            pred_probs, _, _, _ = self.model.predict(self.x, return_uncertainties=True)
            pred_probs = pred_probs.numpy()
        else:
            pred_probs = np.asarray(self.model.predict(self.x, return_uncertainties=False))
        logs[f"{self.name}_csi"] = self.mean_csi(pred_probs)
        true_labels = np.argmax(self.y, 1)
        pred_labels = np.argmax(pred_probs, 1)
        prec, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro"
        )
        logs[f"{self.name}_ave_acc"] = self.ave_acc(true_labels, pred_labels)
        logs[f"{self.name}_prec"] = prec
        logs[f"{self.name}_recall"] = recall
        logs[f"{self.name}_f1"] = f1
        logs[f"{self.name}_auc"] = roc_auc_score(self.y, pred_probs, multi_class='ovr')
        return

    def mean_csi(self, pred_probs):
        pred_labels = np.argmax(pred_probs, 1)
        confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
        rocs = []
        for i in range(pred_probs.shape[1]):
            forecasts = confidences.copy()
            obs = np.where(np.argmax(self.y, 1) == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts[:, 0], obs)
            rocs.append(roc.max_csi())
        return np.mean(rocs)

    def ave_acc(self, true_labels, pred_labels):
        return np.mean(
            [
                (
                        true_labels[np.where(true_labels == _label)]
                        == pred_labels[np.where(true_labels == _label)]
                ).mean()
                for _label in np.unique(true_labels)
            ]
        )

    def mce(self, true_labels, pred_probs):
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels

        mce = 0.0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower).astype(float) * (
                    confidences <= bin_upper
            ).astype(float)
            prop_in_bin = in_bin.astype(float).mean()
            in_bin = in_bin.squeeze(-1).astype(int)
            if prop_in_bin > 0:
                try:
                    max_accuracy_in_bin = accuracies[in_bin].astype(float).max()
                    max_confidence_in_bin = confidences[in_bin].max()
                    max_calibration = np.abs(max_confidence_in_bin - max_accuracy_in_bin)
                    mce = max(mce, max_calibration)
                except:
                    pass

        if mce == 0.0:
            return self.bin_lowers.shape[0]
        else:
            return mce

    def ece(self, true_labels, pred_probs):
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels
        ece = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower).astype(float) * (
                    confidences <= bin_upper
            ).astype(float)
            prop_in_bin = in_bin.astype(float).mean()
            in_bin = in_bin.squeeze(-1).astype(int)
            if prop_in_bin > 0:
                try:
                    accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    avg_calibration = (
                            np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    )
                    ece.append(avg_calibration)
                except:
                    pass
        mean = np.mean(ece) if np.isfinite(np.mean(ece)) else self.bin_lowers.shape[0]
        return mean
