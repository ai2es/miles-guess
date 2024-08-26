import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC


class MetricsCalculator:
    """A class to calculate various metrics for model evaluation, including CSI, average accuracy,
    precision, recall, F1 score, AUC, MCE, and ECE.

    Args:
        n_bins (int, optional): Number of bins for MCE and ECE calculations. Default is 10.
        use_uncertainty (bool, optional): If True, the model's uncertainty is used in predictions. Default is False.

    Methods:
        __call__(y_true, y_pred, split="train"):
            Computes and returns a dictionary of metrics based on the true and predicted values.

        mean_csi(y, pred_probs):
            Computes the mean Critical Success Index (CSI) for the predicted probabilities.

        ave_acc(true_labels, pred_labels):
            Computes the average accuracy for the true and predicted labels.

        mce(true_labels, pred_probs):
            Computes the Maximum Calibration Error (MCE) for the predicted probabilities.

        ece(true_labels, pred_probs):
            Computes the Expected Calibration Error (ECE) for the predicted probabilities.
    """

    def __init__(self, n_bins=10, use_uncertainty=False):
        """Initializes the MetricsCalculator with the specified number of bins and uncertainty flag.

        Args:
            n_bins (int, optional): Number of bins for MCE and ECE calculations. Default is 10.
            use_uncertainty (bool, optional): If True, the model's uncertainty is used in predictions. Default is False.
        """
        self.use_uncertainty = use_uncertainty
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_true, y_pred, split="train"):
        """Computes various metrics based on the true and predicted values.

        Args:
            y_true (torch.Tensor): Tensor of true labels (one-hot encoded).
            y_pred (torch.Tensor): Tensor of predicted probabilities.
            split (str, optional): Indicates which data split is being evaluated (e.g., "train", "val"). Default is "train".

        Returns:
            dict: A dictionary containing the computed metrics, including CSI, average accuracy, precision, recall,
                  F1 score, and AUC.
        """
        logs = {}

        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        true_labels = np.argmax(y_true, axis=1)
        pred_probs = y_pred if self.use_uncertainty else y_pred
        pred_labels = np.argmax(pred_probs, axis=1)

        logs[f"{split}_csi"] = self.mean_csi(y_true, pred_probs)
        prec, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro"
        )
        logs[f"{split}_ave_acc"] = self.ave_acc(true_labels, pred_labels)
        logs[f"{split}_prec"] = prec
        logs[f"{split}_recall"] = recall
        logs[f"{split}_f1"] = f1
        try:
            logs[f"{split}_auc"] = roc_auc_score(y_true, pred_probs, multi_class='ovr')
        except ValueError:
            # AUC computation might fail if there's only one class in either true labels or predicted labels
            logs[f"{split}_auc"] = None

        return logs

    def mean_csi(self, y, pred_probs):
        """Computes the mean Critical Success Index (CSI) for the predicted probabilities.

        Args:
            y (numpy.ndarray): Array of true labels (one-hot encoded).
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The mean Critical Success Index.
        """
        pred_labels = np.argmax(pred_probs, 1)
        confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
        rocs = []
        for i in range(pred_probs.shape[1]):
            forecasts = confidences.copy()
            obs = np.where(np.argmax(y, 1) == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts[:, 0], obs)
            rocs.append(roc.max_csi())
        return np.mean(rocs)

    def ave_acc(self, true_labels, pred_labels):
        """Computes the average accuracy for the true and predicted labels.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_labels (numpy.ndarray): Array of predicted labels.

        Returns:
            float: The average accuracy.
        """
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
        """Computes the Maximum Calibration Error (MCE) for the predicted probabilities.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The Maximum Calibration Error.
        """
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels

        mce = 0.0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
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
                except ValueError:
                    pass

        return mce if mce != 0.0 else self.bin_lowers.shape[0]

    def ece(self, true_labels, pred_probs):
        """Computes the Expected Calibration Error (ECE) for the predicted probabilities.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The Expected Calibration Error.
        """
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels
        ece = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
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
                except ValueError:
                    pass
        mean_ece = np.mean(ece) if np.isfinite(np.mean(ece)) else self.bin_lowers.shape[0]
        return mean_ece
