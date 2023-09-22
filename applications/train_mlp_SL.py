import logging
from echo.src.base_objective import BaseObjective
import copy
import yaml
import sys
import os
import gc
import pickle
import optuna
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser

from keras import backend as K
from mlguess.keras.models import BaseRegressor as RegressorDNN
from mlguess.keras.callbacks import get_callbacks
from mlguess.splitting import load_splitter
from mlguess.regression_uq import compute_results
from mlguess.preprocessing import load_preprocessing
from mlguess.keras.seed import seed_everything
from mlguess.pbs import launch_pbs_jobs
from bridgescaler import save_scaler
from collections import defaultdict
from mlguess.regression_metrics import regression_metrics


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        K.clear_session()
        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]
        # Only use 1 split
        conf["data"]["n_splits"] = 1
        conf["ensemble"]["n_models"] = 1
        conf["ensemble"]["monte_carlo_passes"] = 0

        try:
            return trainer(conf, trial=trial)
        except Exception as E:
            logger.warning(f"Trial {trial.number} failed due to error {str(E)}")
            raise optuna.TrialPruned()


def trainer(conf, trial=False, mode="single"):
    # load seed from the config and set globally
    seed = conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    data_params = conf["data"]
    training_metric = conf["training_metric"]

    # ensemble parameters
    monte_carlo_passes = conf["ensemble"]["monte_carlo_passes"]
    n_models = conf["ensemble"]["n_models"]
    n_splits = conf["ensemble"]["n_splits"]

    model_params = conf["model"]
    model_params["save_path"] = os.path.join(save_loc, "models")

    # Load data, splitter, and scalers
    data = pd.read_csv(conf["data"]["save_loc"])
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])

    split_col = data_params["split_col"]
    input_cols = data_params["input_cols"]
    output_cols = data_params["output_cols"]

    if trial is False:  # Create directories if ECHO is NOT running
        os.makedirs(os.path.join(save_loc, mode), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/models"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/metrics"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/evaluate"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/scalers"), exist_ok=True)
        # Update where the best model will be saved
        # conf["save_loc"] = os.path.join(save_loc, f"{mode}", "models")
        conf["model"]["save_path"] = os.path.join(save_loc, f"{mode}", "models")
        conf["model"]["model_name"] = "best.h5"

        if not os.path.isfile(os.path.join(save_loc, f"{mode}", "models", "model.yml")):
            with open(
                os.path.join(save_loc, f"{mode}/models", "model.yml"), "w"
            ) as fid:
                yaml.dump(conf, fid)

    # Need the same test_data for all trained models (data and model ensembles)
    gsp = load_splitter(
        data_params["splitter"],
        n_splits=1,
        random_state=seed,
        train_size=data_params["train_size"],
    )
    splits = list(gsp.split(data, groups=data[split_col]))
    train_index, test_index = splits[0]
    _train_data, _test_data = (
        data.iloc[train_index].copy(),
        data.iloc[test_index].copy(),
    )

    # Save arrays for ensembles
    if n_models > 1:
        ensemble_mu = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
        ensemble_var = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
    else:
        ensemble_mu = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
        ensemble_var = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

    best_model = None
    # best_data_split = None
    best_model_score = 1e10
    
    results_dict = defaultdict(list)
    ensemble_dict = defaultdict(list)
    
    # Train ensemble of deep models
    _ensemble_pred_deep = np.zeros((n_models, _test_data.shape[0], len(output_cols)))

    for model_seed in range(n_models):

        # Make N train-valid splits using day as grouping variable
        gsp = load_splitter(
            data_params["splitter"],
            n_splits=n_splits,
            random_state=seed,
            train_size=0.885,
        )
        splits = list(gsp.split(_train_data, groups=_train_data[split_col]))

        # Train ensemble of CV-split models
        _ensemble_pred = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

        if n_models > 1:
            # If only looping over data splits, the model is called below
            # Otherwise we call it here so it can be copied (same random seed)
            _model = RegressorDNN(**model_params)
            # build the model here so the weights are initialized (and can be copied below)
            _model.build_neural_network(
                _train_data[input_cols].shape[-1], _train_data[output_cols].shape[-1]
            )

        # Create ensemble from n_splits number of data splits
        for data_seed in range(n_splits):

            # select indices from the split, data splits
            train_index, valid_index = splits[data_seed]
            test_data = copy.deepcopy(_test_data)
            train_data, valid_data = (
                _train_data.iloc[train_index].copy(),
                _train_data.iloc[valid_index].copy(),
            )
            # preprocess x-transformations
            x_scaler, y_scaler = load_preprocessing(conf, seed=seed)
            if x_scaler:
                x_train = x_scaler.fit_transform(train_data[input_cols])
                x_valid = x_scaler.transform(valid_data[input_cols])
                x_test = x_scaler.transform(test_data[input_cols])
            else:
                x_train = train_data[input_cols].values
                x_valid = valid_data[input_cols].values
                x_test = test_data[input_cols].values

            # preprocess y-transformations
            if y_scaler:
                y_train = y_scaler.fit_transform(train_data[output_cols])
                y_valid = y_scaler.transform(valid_data[output_cols])
                y_test = y_scaler.transform(test_data[output_cols])
            else:
                y_train = train_data[output_cols].values
                y_valid = valid_data[output_cols].values
                y_test = test_data[output_cols].values

            # duplicate the model (same seed)
            model = RegressorDNN(**model_params)
            model.build_neural_network(x_train.shape[-1], y_train.shape[-1])
            if n_models > 1:  # duplicate the model (same seed)
                model.model.set_weights(_model.model.get_weights())

            # fit the model
            model.fit(
                x_train,
                y_train,
                validation_data=(x_valid, y_valid),
                callbacks=get_callbacks(conf, path_extend=f"{mode}/models"),
            )
            history = model.model.history
            
            # predict/evaluate on the validation set 
            mu = model.predict(x_valid, y_scaler)
            val_metrics = regression_metrics(y_scaler.inverse_transform(y_valid), mu)
            for k, v in val_metrics.items():
                results_dict[k].append(v)
            
            # predict on the test holdout split
            if n_splits == 1: # If we are only creating a deep ensemble with no CV splits
                _ensemble_pred_deep[model_seed] = model.predict(x_test, y_scaler)
            else:
                _ensemble_pred[data_seed] = model.predict(x_test, y_scaler)

            # If ECHO is running this script, n_splits has been set to 1, return the metric here
            if trial is not False:
                return val_metrics

            # Save model weights
            model.model_name = f"model_seed{model_seed}_split{data_seed}.h5"
            model.save_model()
            
            # Save scalers
            for scaler_name, scaler in zip(
                ["input", "output"], [x_scaler, y_scaler]
            ):
                fn = os.path.join(
                    save_loc, f"{mode}/scalers", f"{scaler_name}_seed{model_seed}_split{data_seed}.json"
                )
                try:
                    save_scaler(scaler, fn)
                except TypeError:
                    with open(fn, "wb") as fid:
                        pickle.dump(scaler, fid)
            
            # Save if its the best model
            if min(history.history[training_metric]) < best_model_score:
                best_model = model
                # Break the current symlink
                if os.path.isfile(os.path.join(save_loc, mode, "models", "best.h5")):
                    os.remove(os.path.join(save_loc, mode, "models", "best.h5"))
                    os.remove(os.path.join(save_loc, mode, "models", "best_training_var.txt"))
                
                ensemble_name = f"model_seed{model_seed}_split{data_seed}"
                os.symlink(
                    os.path.join(save_loc, mode, "models", f"{ensemble_name}.h5"),
                    os.path.join(save_loc, mode, "models", "best.h5"),
                )
                os.symlink(
                    os.path.join(save_loc, mode, "models", f"{ensemble_name}_training_var.txt"),
                    os.path.join(save_loc, mode, "models", "best_training_var.txt"),
                )
                #Save scalers
                for scaler_name in ["input", "output"]:
                    if os.path.isfile(os.path.join(save_loc, f"{mode}/scalers", f"best_{scaler_name}.json")):
                        os.remove(os.path.join(save_loc, f"{mode}/scalers", f"best_{scaler_name}.json"))
                    fn1 = os.path.join(
                        save_loc, f"{mode}/scalers", f"{scaler_name}_seed{model_seed}_split{data_seed}.json"
                    )
                    fn2 = os.path.join(
                        save_loc, f"{mode}/scalers", f"best_{scaler_name}.json"
                    )
                    os.symlink(fn1, fn2)

            # evaluate on the test holdout split
            if mode == "cv_ensemble" and monte_carlo_passes > 0:
                # elif monte_carlo_passes > 0:  # mode = seed or single
                # Create ensemble from MC dropout
                dropout_mu = model.predict_monte_carlo(
                    x_test, y_test, forward_passes=monte_carlo_passes, scaler=y_scaler
                )
                # Calculating mean across multiple MCD forward passes
                # shape (n_samples, n_classes)
                ensemble_mu[data_seed] = np.mean(dropout_mu, axis=0)
                # Calculating variance across multiple MCD forward passes
                # shape (n_samples, n_classes)
                ensemble_var[data_seed] = np.var(dropout_mu, axis=0)
                # Compute metrics
                test_metrics = regression_metrics(
                    y_scaler.inverse_transform(y_test), 
                    ensemble_mu[data_seed], 
                    total=np.sqrt(ensemble_var[data_seed]))
                for k, v in test_metrics.items():
                    ensemble_dict[k].append(v)
                    if k in results_dict:
                        results_dict[k].append(v)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        if mode == "multi_ensemble":
            # Compute uncertainties for the data ensemble
            ensemble_mu[model_seed] = np.mean(_ensemble_pred, 0)
            ensemble_var[model_seed] = np.var(_ensemble_pred, 0)
            # Compute metrics on the validation and test splits
            test_metrics = regression_metrics(
                y_scaler.inverse_transform(y_test), 
                ensemble_mu[model_seed], 
                total=np.sqrt(ensemble_var[model_seed]))
            for k, v in test_metrics.items():
                ensemble_dict[k].append(v)
                #if k in results_dict:
                #    results_dict[k].append(v)

        elif mode == "deep_ensemble" and monte_carlo_passes > 0:
            # elif monte_carlo_passes > 0:  # mode = seed or single
            # Create ensemble from MC dropout
            dropout_mu = best_model.predict_monte_carlo(
                x_test, y_test, forward_passes=monte_carlo_passes, scaler=y_scaler
            )
            # Calculating mean across multiple MCD forward passes
            ensemble_mu[model_seed] = np.mean(
                dropout_mu, axis=0
            )  # shape (n_samples, n_classes)
            # Calculating variance across multiple MCD forward passes
            ensemble_var[model_seed] = np.var(
                dropout_mu, axis=0
            )  # shape (n_samples, n_classes)
            # Compute metrics on the validation and test splits
            test_metrics = regression_metrics(
                y_scaler.inverse_transform(y_test), 
                ensemble_mu[model_seed], 
                total=np.sqrt(ensemble_var[model_seed]))
            for k, v in test_metrics.items():
                ensemble_dict[k].append(v)
                #if k in results_dict:
                #    results_dict[k].append(v)

        elif mode == "single":  # mode = seed or single
            ensemble_mu[model_seed] = best_model.predict(x_test, y_scaler)
            # Compute metrics on validation set 
            test_metrics = regression_metrics(
                y_scaler.inverse_transform(y_test), 
                ensemble_mu[model_seed])
            for k, v in test_metrics.items():
                results_dict[k].append(v)
            if monte_carlo_passes > 0:
                pass  # Add this 

    #  Save dictionary containing all model training metrics
    pd.DataFrame.from_dict(results_dict).to_csv(
        os.path.join(save_loc, f"{mode}/evaluate/model_metrics.csv"))
    #  If we have not created an ensemble, we are finished.
    if mode == "single":
        _test_data[[f"{x}_pred" for x in output_cols]] = ensemble_mu[0]
        _test_data.to_csv(os.path.join(save_loc, f"{mode}/evaluate", "test.csv"))
        return 1.0

    # We only created ensemble over model or seed but not both and no MC dropout
    if mode in ["cv_ensemble", "deep_ensemble"]:
        if n_splits == 1:
            mu = np.mean(_ensemble_pred_deep, 0)
            var = np.var(_ensemble_pred_deep, 0)
        else:
            mu = np.mean(_ensemble_pred, 0)
            var = np.var(_ensemble_pred, 0)
        # Save the ensemble metrics
        ensemble_metrics = regression_metrics(
            y_scaler.inverse_transform(y_test), 
            mu,
            total=np.sqrt(var))
        ensemble_metrics = {k: [v] for k,v in ensemble_metrics.items()}
        pd.DataFrame.from_dict(ensemble_metrics).to_csv(
            os.path.join(save_loc, f"{mode}/evaluate/ensemble_metrics.csv"))
        if monte_carlo_passes > 0:
            pd.DataFrame.from_dict(ensemble_dict).to_csv(
                os.path.join(save_loc, f"{mode}/evaluate/mc_ensemble_metrics.csv"))
        else:
            _test_data[[f"{x}_ensemble_pred" for x in output_cols]] = mu
            _test_data[[f"{x}_ensemble_var" for x in output_cols]] = var
            _test_data.to_csv(os.path.join(save_loc, f"{mode}/evaluate", "test.csv"))
            return 1.0

    # Compute aleatoric and epistemic uncertainties using law of total uncertainty
    ensemble_mean = np.mean(ensemble_mu, 0)
    ensemble_aleatoric = np.mean(ensemble_var, 0)
    ensemble_epistemic = np.var(ensemble_mu, 0)
    total = np.sqrt(ensemble_aleatoric+ensemble_epistemic)
    lotv_metrics = regression_metrics(
        y_scaler.inverse_transform(y_test), 
        ensemble_mean, 
        total=total)
    # Save metrics
    lotv_metrics = {k: [v] for k,v in lotv_metrics.items()}
    pd.DataFrame.from_dict(lotv_metrics).to_csv(
        os.path.join(save_loc, f"{mode}/evaluate/lotv_metrics.csv"))

    # add to df and save
    _test_data[[f"{x}_ensemble_pred" for x in output_cols]] = ensemble_mean
    _test_data[[f"{x}_ensemble_ale" for x in output_cols]] = ensemble_aleatoric
    _test_data[[f"{x}_ensemble_epi" for x in output_cols]] = ensemble_epistemic

    # make some figures
    _test_data.to_csv(os.path.join(save_loc, f"{mode}/evaluate", "test.csv"))
    compute_results(
        _test_data,
        output_cols,
        ensemble_mean,
        ensemble_aleatoric,
        ensemble_epistemic,
        fn=os.path.join(save_loc, f"{mode}", "metrics"),
    )

    return 1.0


if __name__ == "__main__":

    description = "Train an MLP regression model on a surface layer data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit 1 worker to PBS.",
    )
    args_dict = vars(parser.parse_args())
    config = args_dict.pop("model_config")
    launch = bool(int(args_dict.pop("launch")))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)

    # Load the "multi_ensemble" details fron the config
    n_models = conf["ensemble"]["n_models"]
    n_splits = conf["ensemble"]["n_splits"]
    monte_carlo_passes = conf["ensemble"]["monte_carlo_passes"]

    # How is this script supposed to run?
    if n_splits > 1 and n_models == 1:
        mode = "cv_ensemble"
    elif n_splits == 1 and n_models > 1:
        mode = "deep_ensemble"
    elif n_splits == 1 and n_models == 1:
        mode = "single"
    elif n_splits > 1 and n_models > 1:
        mode = "multi_ensemble"
    else:
        raise ValueError(
            "Incorrect selection of n_splits or n_models. Both must be at greater than or equal to 1."
        )
    logger.info(
        f"Running with ensemble mode: {mode} with n_splits: {n_splits}, n_models: {n_models}, mc_steps: {monte_carlo_passes}"
    )

    if launch:
        from pathlib import Path

        script_path = Path(__file__).absolute()
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, script_path)
        sys.exit()

    result = trainer(conf, mode=mode)
