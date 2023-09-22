import logging
import tqdm

from collections import defaultdict
from echo.src.base_objective import BaseObjective
import copy
import yaml
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
from mlguess.pit import pit_deviation_skill_score, pit_deviation
from mlguess.keras.models import EvidentialRegressorDNN
from mlguess.keras.callbacks import get_callbacks
from mlguess.splitting import load_splitter
from mlguess.regression_uq import compute_results
from mlguess.preprocessing import load_preprocessing
from mlguess.keras.seed import seed_everything
from mlguess.regression_metrics import regression_metrics
from mlguess.pbs import launch_pbs_jobs
from bridgescaler import save_scaler


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
    
    if trial is False:  # Dont create directories if ECHO is running
        os.makedirs(os.path.join(save_loc, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, "scalers"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, "evaluate"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, "metrics"), exist_ok=True)
        conf["model"]["save_path"] = os.path.join(save_loc, "models")
        
        if not os.path.isfile(os.path.join(save_loc, "models", "model.yml")):
            with open(
                os.path.join(save_loc, "models", "model.yml"), "w"
            ) as fid:
                yaml.dump(conf, fid)

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
    best_model_score = 1e10 if direction == "min" else -1e10
    results_dict = defaultdict(list)
    
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
            callbacks=get_callbacks(conf, path_extend="models"),
        )
        history = model.model.history
        
        ####################
        #
        # VALIDATE THE MODEL
        #
        ####################

        # Compute metrics on validation set 
        mu, ale, epi = model.predict_uncertainty(x_valid)
        total = np.sqrt(ale + epi)
        val_metrics = regression_metrics(y_scaler.inverse_transform(y_valid), mu, total)
        for k, v in val_metrics.items():
            results_dict[k].append(v)
        optimization_metric = val_metrics[training_metric]

        # If ECHO is running this script, n_splits has been set to 1, return the metric here
        if trial is not False:
            for metric in ["val_r2", "val_rmse_ss", "val_crps_ss"]:
                if val_metrics[metric] < 0.0:  # ECHO maxing out negative numbers? Not sure why ... 
                    val_metrics[metric] = 0.0
            return val_metrics
        
        # Write to the logger
        logger.info(
            f"Finished split {data_seed} with metric {training_metric} = {optimization_metric}"
        )

        ##########
        #
        # SAVE MODEL
        #
        ########## 

        # Save model weights
        model.model_name = f"models/model_split{data_seed}.h5"
        model.save_model()

        if conf["ensemble"]["n_splits"] > 1 or conf["ensemble"]["n_models"] > 1:
            pd_history = pd.DataFrame.from_dict(history.history)
            pd_history["data_split"] = data_seed
            pd_history.to_csv(
                os.path.join(conf["save_loc"], "models", f"training_log_split{data_seed}.csv")
            )

        # Save scalers
        for scaler_name, scaler in zip(
            ["input", "output"], [x_scaler, y_scaler]
        ):
            fn = os.path.join(
                save_loc, "scalers", f"{scaler_name}_split{data_seed}.json"
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

            # Break the current symlink
            if os.path.isfile(os.path.join(save_loc, "models", "best.h5")):
                os.remove(os.path.join(save_loc, "models", "best.h5"))
                os.remove(os.path.join(save_loc, "models", "best_training_var.txt"))

            ensemble_name = f"model_split{data_seed}"
            os.symlink(
                os.path.join(save_loc, "models", f"{ensemble_name}.h5"),
                os.path.join(save_loc, "models", "best.h5"),
            )
            os.symlink(
                os.path.join(save_loc, "models", f"{ensemble_name}_training_var.txt"),
                os.path.join(save_loc, "models", "best_training_var.txt"),
            )
            #Save scalers
            for scaler_name in ["input", "output"]:
                if os.path.isfile(os.path.join(save_loc, "scalers", f"best_{scaler_name}.json")):
                    os.remove(os.path.join(save_loc, "scalers", f"best_{scaler_name}.json"))
                fn1 = os.path.join(
                    save_loc, "scalers", f"{scaler_name}_split{data_seed}.json"
                )
                fn2 = os.path.join(
                    save_loc, "scalers", f"best_{scaler_name}.json"
                )
                os.symlink(fn1, fn2)
                                
        ################
        #
        # TEST THE MODEL
        #
        ################

        # evaluate on the test holdout split
        mu, aleatoric, epistemic = model.predict_uncertainty(x_test, scaler=y_scaler)
        total = np.sqrt(aleatoric + epistemic)
        ensemble_mu[data_seed] = mu
        ensemble_ale[data_seed] = aleatoric
        ensemble_epi[data_seed] = epistemic
        test_metrics = regression_metrics(y_scaler.inverse_transform(y_test), mu, total, split="test")
        for k, v in test_metrics.items():
            results_dict[k].append(v)

        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
    # Use the best model and predict on the three splits
    X = [x_train, x_valid, x_test]
    Y = [y_train, y_valid, y_test]
    splits = ["train", "val", "test"]  
    dfs = [train_data, valid_data, test_data]
    
    best_metrics = {}
    for (x, y, split, df) in zip(X, Y, splits, dfs):
        result = best_model.predict_uncertainty(x, scaler=y_scaler)
        mu, aleatoric, epistemic = result
        total = np.sqrt(aleatoric + epistemic)
        for k, v in regression_metrics(y_scaler.inverse_transform(y), mu, total, split=split).items():
            best_metrics[k] = v
        df[[f"{x}_pred" for x in output_cols]] = mu
        df[[f"{x}_ale" for x in output_cols]] = aleatoric
        df[[f"{x}_epi" for x in output_cols]] = epistemic
        df.to_csv(os.path.join(save_loc, "evaluate", f"{split}.csv"))

    # Save the test ensemble as numpy array
    np.save(os.path.join(save_loc, "evaluate/test_mu.npy"), ensemble_mu)
    np.save(os.path.join(save_loc, "evaluate/test_aleatoric.npy"), ensemble_ale)
    np.save(os.path.join(save_loc, "evaluate/test_epistemic.npy"), ensemble_epi)

    # Save metrics
    pd.DataFrame.from_dict(results_dict).to_csv(
        os.path.join(save_loc, "evaluate/ensemble_metrics.csv"))
    pd.DataFrame.from_dict({
        "metric": list(best_metrics.keys()),
        "value": list(best_metrics.values())
    }).to_csv(os.path.join(save_loc, "evaluate/best_metrics.csv"))

    # make some figures
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

    if launch:
        from pathlib import Path
        script_path = Path(__file__).absolute()
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, script_path)
        sys.exit()

    result = trainer(conf)
