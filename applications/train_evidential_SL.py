import logging
import tqdm

from collections import defaultdict
from echo.src.base_objective import BaseObjective
import copy
import yaml
import shutil
import sys
import os
import gc
import warnings
import optuna
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser

from tensorflow.keras import backend as K
from evml.pit import pit_deviation_skill_score, pit_deviation
from evml.keras.models import EvidentialRegressorDNN
from evml.keras.callbacks import get_callbacks
from evml.splitting import load_splitter
from evml.regression_uq import compute_results
from evml.preprocessing import load_preprocessing
from evml.keras.seed import seed_everything
from evml.pbs import launch_pbs_jobs
from bridgescaler import save_scaler
from sklearn.metrics import r2_score, mean_squared_error


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
        # Only use 1 data split
        conf["ensemble"]["n_splits"] = 1
        conf["ensemble"]["n_models"] = 1

        try:
            return trainer(conf, trial=trial)
        except Exception as E:
            logger.warning(f"Trial {trial.number} failed due to error {str(E)}")
            raise optuna.TrialPruned()


def trainer(conf, trial=False):
    # load seed from the config and set globally
    seed = conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    data_params = conf["data"]
    training_metric = conf["training_metric"]
    direction = conf["direction"]
    n_splits = conf["ensemble"]["n_splits"]

    model_params = conf["model"]
    model_params["save_path"] = save_loc

    # Load data, splitter, and scalers
    data = pd.read_csv(conf["data"]["save_loc"])
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])

    split_col = data_params["split_col"]
    input_cols = data_params["input_cols"]
    output_cols = data_params["output_cols"]

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
    ensemble_mu = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
    ensemble_ale = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
    ensemble_epi = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

    best_model = None
    best_split = None
    best_model_score = 1e10 if direction == "min" else -1e10
    pitd_dict = defaultdict(list)

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

        # load the model
        model = EvidentialRegressorDNN(**model_params)
        model.build_neural_network(x_train.shape[-1], y_train.shape[-1])
        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            callbacks=get_callbacks(conf, path_extend=""),
        )
        history = model.model.history

        # Get the value of the metric
        if "pit" in training_metric:
            _pitd = []
            mu, ale, epi = model.predict(x_valid)
            for i, col in enumerate(output_cols):
                _pitd.append(
                    pit_deviation_skill_score(
                        y_valid[:, i],
                        np.stack([mu[:, i], np.sqrt(ale[:, i] + epi[:, i])], -1),
                        pred_type="gaussian",
                    )
                )
                # _pitd.append(
                #     pit_deviation_skill_score(
                #         y_valid[:, i],
                #         np.stack([mu[:, i], np.sqrt(ale[:, i])], -1),
                #         pred_type="gaussian",
                #     )
                # )
                # _pitd.append(
                #     pit_deviation_skill_score(
                #         y_valid[:, i],
                #         np.stack([mu[:, i], np.sqrt(epi[:, i])], -1),
                #         pred_type="gaussian",
                #     )
                # )
            optimization_metric = np.mean(_pitd)
        elif "R2" in training_metric:
            mu, ale, epi = model.predict(x_valid)
            rmse = (y_valid - mu) ** 2
            spread = ale + epi
            optimization_metric = r2_score(rmse, spread)
        elif direction == "min":
            optimization_metric = min(history.history[training_metric])
        elif direction == "max":
            optimization_metric = max(history.history[training_metric])

        # If ECHO is running this script, n_splits has been set to 1, return the metric here
        if trial is not False:
            return {
                training_metric: optimization_metric,
                "val_mae": min(history.history["val_mae"]),
            }

        # Write to the logger
        logger.info(
            f"Finished split {data_seed} with metric {training_metric} = {optimization_metric}"
        )

        # Save if its the best model
        c1 = (direction == "min") and (optimization_metric < best_model_score)
        c2 = (direction == "max") and (optimization_metric > best_model_score)
        if c1 | c2:
            best_model = model
            best_model_score = optimization_metric
            best_split = data_seed
            model.model_name = "best.h5"
            model.save_model()
            
            # Save scalers
            for scaler_name, scaler in zip(
                ["input", "output"], [x_scaler, y_scaler]
            ):
                fn = os.path.join(
                    conf["model"]["save_path"], f"{scaler_name}.json"
                )
                try:
                    save_scaler(scaler, fn)
                except TypeError:
                    with open(fn, "wb") as fid:
                        pickle.dump(scaler, fid)

        # evaluate on the test holdout split
        result = model.predict(x_test, scaler=y_scaler)
        mu, aleatoric, epistemic = result
        ensemble_mu[data_seed] = mu
        ensemble_ale[data_seed] = aleatoric
        ensemble_epi[data_seed] = epistemic

        for i, col in enumerate(output_cols):
            pitd_dict[col].append(
                pit_deviation_skill_score(
                    test_data[output_cols].values[:, i],
                    np.stack(
                        [mu[:, i], np.sqrt(aleatoric[:, i] + epistemic[:, i])], -1
                    ),
                    pred_type="gaussian",
                )
            )

        # check if this is the best model
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Compute uncertainties
    mu = ensemble_mu[best_split]
    epistemic = ensemble_epi[best_split]
    aleatoric = ensemble_ale[best_split]

    # add to df and save
    _test_data[[f"{x}_pred" for x in output_cols]] = mu
    _test_data[[f"{x}_ale" for x in output_cols]] = aleatoric
    _test_data[[f"{x}_epi" for x in output_cols]] = epistemic

    os.makedirs(os.path.join(save_loc, "evaluate"), exist_ok=True)
    _test_data.to_csv(os.path.join(save_loc, "evaluate/test.csv"))
    np.save(os.path.join(save_loc, "evaluate/test_mu.npy"), ensemble_mu)
    np.save(os.path.join(save_loc, "evaluate/test_aleatoric.npy"), ensemble_ale)
    np.save(os.path.join(save_loc, "evaluate/test_epistemic.npy"), ensemble_epi)

    # Save PITD
    pd.DataFrame.from_dict(pitd_dict).to_csv(os.path.join(save_loc, "evaluate/pitd.csv"))

    # make some figures
    os.makedirs(os.path.join(save_loc, "metrics"), exist_ok=True)
    compute_results(
        _test_data,
        output_cols,
        mu,
        aleatoric,
        epistemic,
        ensemble_mu=ensemble_mu,
        ensemble_type="Cross validation ensemble",
        fn=os.path.join(save_loc, "metrics"),
    )


if __name__ == "__main__":

    description = "Train an evidential model on a surface layer data set"
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

    conf["model"]["save_path"] = save_loc
    conf["model"]["model_name"] = "best.h5"

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    if launch:
        from pathlib import Path
        script_path = Path(__file__).absolute()
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, script_path)
        sys.exit()

    result = trainer(conf)
