import logging
import tqdm

from echo.src.base_objective import BaseObjective
import pickle
import copy
import yaml
import sys
import os
import gc
import optuna
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser
from collections import defaultdict
from bridgescaler import save_scaler

from keras import backend as K
from evml.pit import pit_deviation_skill_score, pit_deviation
from evml.keras.model_refactor import GaussianRegressorDNN
from evml.keras.callbacks import get_callbacks
from evml.splitting import load_splitter
from evml.regression_uq import compute_results
from evml.preprocessing import load_preprocessing
from evml.keras.seed import seed_everything
from evml.pbs import launch_pbs_jobs
import traceback


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
        # Only use 1 data split and 1 model (one model seed)
        conf["ensemble"]["n_splits"] = 1
        conf["ensemble"]["n_models"] = 1
        # conf["ensemble"]["monte_carlo_passes"] = 0

        try:
            return trainer(conf, trial=trial)
        except Exception as E:
            logger.warning(f"Trial {trial.number} failed due to error {str(E)}")
            print(traceback.format_exc())
            raise optuna.TrialPruned()


def trainer(conf, trial=False, mode="single"):
    # load seed from the config and set globally
    seed = conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    data_params = conf["data"]
    training_metric = conf["training_metric"]
    direction = conf["direction"]

    # ensemble parameters
    monte_carlo_passes = conf["ensemble"]["monte_carlo_passes"]
    n_models = conf["ensemble"]["n_models"]
    n_splits = conf["ensemble"]["n_splits"]

    model_params = conf["model"]
    model_params["save_path"] = save_loc

    # Load data, splitter, and scalers
    data = pd.read_csv(conf["data"]["save_loc"])
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])

    split_col = data_params["split_col"]
    input_cols = data_params["input_cols"]
    output_cols = data_params["output_cols"]

    # Make some directories if ECHO is not running
    if monte_carlo_passes > 1 and trial is False:
        os.makedirs(os.path.join(save_loc, "monte_carlo/metrics"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, "monte_carlo/evaluate"), exist_ok=True)
    if trial is False:  # Dont create directories if ECHO is running
        os.makedirs(os.path.join(save_loc, mode), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/models"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/metrics"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/evaluate"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{mode}/scalers"), exist_ok=True)
        # Update where the best model will be saved
        # conf["save_loc"] = os.path.join(save_loc, f"{mode}/models")
        conf["model"]["save_path"] = os.path.join(save_loc, f"{mode}/models")
        conf["model"]["model_name"] = "best.h5"

        if not os.path.isfile(os.path.join(save_loc, f"{mode}/models", "model.yml")):
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

    # Make N train-valid splits using day as grouping variable
    gsp = load_splitter(
        data_params["splitter"], n_splits=n_splits, random_state=seed, train_size=0.885
    )
    splits = list(gsp.split(_train_data, groups=_train_data[split_col]))

    # Train ensemble of parametric models
    if mode == "seed":
        ensemble_mu = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
        ensemble_var = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
    else:
        ensemble_mu = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
        ensemble_var = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

    best_model = None
    best_data_split = None
    best_model_score = 1e10 if direction == "min" else -1e10

    for model_seed in range(n_models):

        # Make N train-valid splits using day as grouping variable
        gsp = load_splitter(
            data_params["splitter"],
            n_splits=n_splits,
            random_state=seed,
            train_size=0.885,  # move this to the config
        )
        splits = list(gsp.split(_train_data, groups=_train_data[split_col]))

        if n_models > 1:
            # If only looping over data splits, the model is called below
            # Otherwise we call it here so it can be copied (same random seed)
            _model = GaussianRegressorDNN(**model_params)
            # build the model here so the weights are initialized (and can be copied below)
            _model.build_neural_network(
                _train_data[input_cols].shape[-1], _train_data[output_cols].shape[-1]
            )

        # Loop over data splits and train models
        for data_seed in tqdm.tqdm(range(n_splits)):

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

            # Copy / initialize model
            model = GaussianRegressorDNN(**model_params)
            model.build_neural_network(x_train.shape[-1], y_train.shape[-1])
            if (
                n_models > 1
            ):  # duplicate the model (same seed) -- but I should check this!
                model.model.set_weights(_model.model.get_weights())

            model.fit(
                x_train,
                y_train,
                validation_data=(x_valid, y_valid),
                callbacks=get_callbacks(conf, path_extend=f"{mode}/models"),
            )
            history = model.model.history

            # Get the value of the metric
            if "pit" in training_metric:
                pitd = []
                mu, var = model.predict_uncertainty(x_valid, y_scaler)
                #mu, var = model.calc_uncertainties(y_pred, y_scaler)
                for i, col in enumerate(output_cols):
                    pitd.append(
                        pit_deviation(
                            y_valid[:, i],
                            np.stack([mu[:, i], np.sqrt(var[:, i])], -1),
                            pred_type="gaussian",
                        )
                    )
                optimization_metric = np.mean(pitd)
            elif direction == "min":
                optimization_metric = min(history.history[training_metric])
            elif direction == "max":
                optimization_metric = max(history.history[training_metric])

            # If ECHO is running this script, n_splits has been set to 1, return the metric here
            if trial is not False and conf["ensemble"]["monte_carlo_passes"] == 0:
                return {
                    training_metric: optimization_metric,
                    "val_mae": min(history.history["val_mae"]),
                }

            # Write to the logger
            logger.info(
                f"Finished model/data split {model_seed}/{data_seed} with metric {training_metric} = {optimization_metric}"
            )
            
            ##########
            #
            # SAVE MODEL
            #
            ########## 
            
            # Save model weights
            model.model_name = f"model_seed{model_seed}_split{data_seed}.h5"
            model.save_model()
            
            if conf["ensemble"]["n_splits"] > 1 or conf["ensemble"]["n_models"] > 1:
                pd_history = pd.DataFrame.from_dict(history.history)
                pd_history["data_split"] = data_seed
                pd_history["model_split"] = model_seed
                pd_history.to_csv(
                    os.path.join(conf["save_loc"], mode, "models", f"training_log_seed{model_seed}_split{data_seed}.csv")
                )
            
            # Save scalers
            for scaler_name, scaler in zip(
                ["input", "output"], [x_scaler, y_scaler]
            ):
                fn = os.path.join(
                    save_loc, f"{mode}/scalers", f"{scaler_name}.json"
                )
                try:
                    save_scaler(scaler, fn)
                except TypeError:
                    with open(fn, "wb") as fid:
                        pickle.dump(scaler, fid)

            # Symlink if its the best model
            c1 = (direction == "min") and (optimization_metric < best_model_score)
            c2 = (direction == "max") and (optimization_metric > best_model_score)
            if c1 | c2:
                best_model = model
                best_model_score = optimization_metric
                best_data_split = data_seed
                model.model_name = "best.h5"
                model.save_model()
                # ensemble_name = f"model_seed{model_seed}_split{data_seed}"
                # os.symlink(
                #     os.path.join(save_loc, mode, "models", f"{ensemble_name}.h5"),
                #     os.path.join(save_loc, mode, "models", "best.h5"),
                # )
                # os.symlink(
                #     os.path.join(save_loc, mode, "models", f"{ensemble_name}_training_var.txt"),
                #     os.path.join(save_loc, mode, "models", "best_training_var.txt"),
                # )
                # Save scalers
                # for scaler_name in ["input", "output"]:
                #     fn1 = os.path.join(
                #         save_loc, f"{mode}/scalers", f"{scaler_name}.json"
                #     )
                #     fn2 = os.path.join(
                #         save_loc, f"{mode}/scalers", f"best_{scaler_name}.json"
                #     )
                #     os.symlink(fn1, fn2)
                for scaler_name, scaler in zip(
                    ["input", "output"], [x_scaler, y_scaler]
                ):
                    fn = os.path.join(
                        save_loc, f"{mode}/scalers", f"best_{scaler_name}.json"
                    )
                    try:
                        save_scaler(scaler, fn)
                    except TypeError:
                        with open(fn, "wb") as fid:
                            pickle.dump(scaler, fid)
                    

            if trial is not False:
                continue

            # Evaluate on the test holdout split
            for split, x_split, df in zip(
                ["test"], [x_test], [test_data]
            ):

                mu, var = model.predict_uncertainty(x_split, y_scaler)

                if mode == "seed":
                    ensemble_mu[model_seed] = mu
                    ensemble_var[model_seed] = var
                else:
                    ensemble_mu[data_seed] = mu
                    ensemble_var[data_seed] = var

                # Save the ensemble member df
                df[[f"{x}_pred" for x in output_cols]] = mu
                df[[f"{x}_ale" for x in output_cols]] = var
                df.to_csv(
                    os.path.join(
                        save_loc, f"{mode}/evaluate", f"{split}_{data_seed}.csv"
                    )
                )

            # Delete old models
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    # Evaluation and calculation of uncertainties
    if mode != "single" and trial is False:
        logger.info(f"Computing uncertainties from the {mode} ensemble")

        # Compute uncertainties for the data/model ensemble
        ensemble_epistemic = np.var(ensemble_mu, axis=0)
        ensemble_aleatoric = np.mean(ensemble_var, axis=0)
        ensemble_mean = np.mean(ensemble_mu, axis=0)

        _test_data[[f"{x}_pred" for x in output_cols]] = ensemble_mean
        _test_data[[f"{x}_ale" for x in output_cols]] = ensemble_aleatoric
        _test_data[[f"{x}_epi" for x in output_cols]] = ensemble_epistemic

        # save
        _test_data.to_csv(os.path.join(save_loc, f"{mode}/evaluate/test.csv"))
        np.save(os.path.join(save_loc, f"{mode}/evaluate/test_mu.npy"), ensemble_mu)
        np.save(os.path.join(save_loc, f"{mode}/evaluate/test_sigma.npy"), ensemble_var)

        # make some figures
        title = "Model seed ensemble" if mode == "seed" else "Cross validation ensemble"
        compute_results(
            _test_data,
            output_cols,
            ensemble_mean,
            ensemble_aleatoric,
            ensemble_epistemic,
            ensemble_mu=ensemble_mu,
            ensemble_type=title,
            fn=os.path.join(save_loc, f"{mode}/metrics"),
        )

    # Compute epistemic uncertainty via MC-dropout
    if monte_carlo_passes > 0:
        logger.info("Computing uncertainties using Monte Carlo dropout")

        if trial is not False:  # If running ECHO, use the valid split
            x = x_valid
            y = y_valid
        else:  # Otherwise use the test split
            x = x_test
            y = y_test

        dropout_mu, dropout_aleatoric = best_model.predict_monte_carlo(
            x,
            y,
            forward_passes=monte_carlo_passes,
            y_scaler=y_scaler,
        )

        # Calculating mean across multiple MCD forward passes
        mc_mu = np.mean(dropout_mu, axis=0)  # shape (n_samples, n_classes)
        mc_aleatoric = np.mean(
            dropout_aleatoric, axis=0
        )  # shape (n_samples, n_classes)
        # Calculating variance across multiple MCD forward passes
        mc_epistemic = np.var(dropout_mu, axis=0)  # shape (n_samples, n_classes)

        # Compute PITD
        pitd_dict = defaultdict(list)
        for i, col in enumerate(output_cols):
            pitd_dict[col].append(
                pit_deviation(
                    y[:, i],
                    np.stack(
                        [mu[:, i], np.sqrt(mc_aleatoric[:, i] + mc_epistemic[:, i])], -1
                    ),
                    pred_type="gaussian",
                )
            )

        if trial is not False:
            optimization_metric = np.mean([x[0] for x in pitd_dict.values()])
            return {
                training_metric: optimization_metric,
                "val_mae": min(history.history["val_mae"]),
            }

        # save
        _test_data[[f"{x}_pred" for x in output_cols]] = mc_mu
        _test_data[[f"{x}_ale" for x in output_cols]] = mc_aleatoric
        _test_data[[f"{x}_epi" for x in output_cols]] = mc_epistemic

        np.save(os.path.join(save_loc, "monte_carlo/evaluate/test_mu.npy"), dropout_mu)
        np.save(
            os.path.join(save_loc, "monte_carlo/evaluate/test_sigma.npy"),
            dropout_aleatoric,
        )
        _test_data.to_csv(os.path.join(save_loc, "monte_carlo/evaluate/test.csv"))
        pd.DataFrame.from_dict(pitd_dict).to_csv(
            os.path.join(save_loc, "monte_carlo/evaluate/pit.csv")
        )

        # Make some figures
        compute_results(
            _test_data,
            output_cols,
            mc_mu,
            mc_aleatoric,
            mc_epistemic,
            ensemble_mu=dropout_mu,
            ensemble_type="Monte Carlo ensemble",
            fn=os.path.join(save_loc, "monte_carlo/metrics"),
        )

    return 1


if __name__ == "__main__":

    description = "Train an parametric regression model on a surface layer data set"
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

    # Load the "ensemble" details fron the config
    # Options: ensemble, seed, monte-carlo, single-model
    # Ensemble if n_splits > 1 and n_models = 1
    # Seed if n_splits = 1 and n_models > 1
    # Single model if n_splits = 1 and n_models = 1
    # Monte-Carlo if mc_steps > 0
    n_models = conf["ensemble"]["n_models"]
    n_splits = conf["ensemble"]["n_splits"]
    monte_carlo_passes = conf["ensemble"]["monte_carlo_passes"]
    modes = []
    if n_splits > 1 and n_models == 1:
        mode = "data"
    elif n_splits == 1 and n_models > 1:
        mode = "seed"
    elif n_splits == 1 and n_models == 1:
        mode = "single"
    else:
        raise ValueError(
            "For the Gaussian model, only one of n_models or n_splits can be > 1 while the other must be 1"
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
