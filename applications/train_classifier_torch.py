import warnings
import os
import sys
import copy
import yaml
import optuna
import shutil
import logging
import importlib
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from echo.src.base_objective import BaseObjective
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from mlguess.torch.distributed import distributed_model_wrapper
from mlguess.torch.scheduler import load_scheduler
from mlguess.pbs import launch_pbs_jobs, launch_distributed_jobs
from mlguess.torch.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO
)
from mlguess.torch.trainer_classifier import Trainer
from mlguess.torch.class_losses import EDLDigammaLoss
from mlguess.torch.models import seed_everything, CategoricalDNN
from mlguess.keras.data import load_ptype_uq, preprocess_data
from mlguess.torch.metrics import MetricsCalculator
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup(rank, world_size, mode):
    """
    Initialize the process group for distributed training.

    Args:
        rank (int): The rank of the current process in the distributed training.
        world_size (int): The total number of processes involved in distributed training.
        mode (str): The mode of training (e.g., 'fsdp' or 'ddp').

    Logs:
        Information about the distributed training setup.
    """

    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def import_class_from_path(class_name, file_path):
    """
    Import a class from a file path.

    Args:
        class_name (str): The name of the class to import.
        file_path (str): The file path of the module containing the class.

    Returns:
        type: The imported class.
    """

    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_dataset_and_sampler(conf, world_size, rank, split, seed=42, return_df=False):
    """
    Load a dataset and its corresponding distributed sampler based on the configuration.

    Args:
        conf (dict): Configuration dictionary specifying dataset and sampler settings.
        world_size (int): The total number of processes involved in distributed training.
        rank (int): The rank of the current process.
        is_train (bool): Whether to load the training or validation dataset.
        seed (int, optional): Random seed for sampling. Defaults to 42.

    Returns:
        tuple: A tuple containing the dataset and the distributed sampler.

    Logs:
        Information about the dataset and sampler loaded.
    """

    # Z-score
    # torch_dataset = import_class_from_path(conf["data"]["dataset_name"], conf["data"]["dataset_path"])
    # dataset = torch_dataset(conf, split='train' if is_train else 'val')

    input_features = []
    for features in ["TEMP_C", "T_DEWPOINT_C", "UGRD_m/s", "VGRD_m/s"]:
        input_features += conf["data"][features]
    output_features = conf["data"]["ptypes"]
    is_train = split == "train"

    # Load data
    _conf = copy.deepcopy(conf)
    _conf.update(conf["data"])
    data = load_ptype_uq(_conf, data_split=0, verbose=1, drop_mixed=False)

    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf["data"] else conf["data"]["scale_groups"]
    groups = [list(conf["data"][g]) for g in scale_groups]
    leftovers = list(
        set(input_features)
        - set([row for group in scale_groups for row in conf["data"][group]])
    )
    if len(leftovers):
        groups.append(leftovers)
    # scale the data
    scaled_data, scalers = preprocess_data(
        data,
        input_features,
        output_features,
        scaler_type=conf["data"]["scaler_type"],
        encoder_type="onehot",
        groups=[],
    )

    X = torch.FloatTensor(scaled_data[f"{split}_x"].values)
    y = torch.LongTensor(np.argmax(scaled_data[f"{split}_y"], axis=-1))

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)

    # Pytorch sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=is_train,
        drop_last=(not is_train)
    )
    logging.info(f"Loaded a {split} torch dataset, and a distributed sampler")

    if return_df:
        return dataset, sampler, data

    return dataset, sampler


def load_model_states_and_optimizer(conf, model, device):
    """
    Load the model, optimizer, scheduler, and scaler states from checkpoint files based on the configuration.

    Args:
        conf (dict): Configuration dictionary specifying the checkpoint paths and training hyperparameters.
        model (torch.nn.Module): The model to load states into.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        tuple: A tuple containing the model, optimizer, scheduler, and scaler.
    """

    # convert $USER to the actual user name
    conf['save_loc'] = save_loc = os.path.expandvars(conf['save_loc'])

    # training hyperparameters
    start_epoch = conf['trainer']['start_epoch']
    learning_rate = float(conf['trainer']['learning_rate'])
    weight_decay = float(conf['trainer']['weight_decay'])
    amp = conf['trainer']['amp']

    # load weights falg
    load_weights = False if 'load_weights' not in conf['trainer'] else conf['trainer']['load_weights']

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if start_epoch == 0 and not load_weights:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # load optimizer and grad scaler states
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        # FSDP checkpoint settings
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))

        else:
            # DDP settings
            if conf["trainer"]["mode"] == "ddp":
                logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Enable updating the lr if not using a policy
    if (conf["trainer"]["update_learning_rate"] if "update_learning_rate" in conf["trainer"] else False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    return model, optimizer, scheduler, scaler


def main(rank, world_size, conf, trial=False):
    """
    Main function to set up and run the training and validation process.

    Args:
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes involved in distributed training.
        conf (dict): Configuration dictionary containing settings for the model, training, and data.
        trial (bool, optional): Flag to indicate if the function is being used in a hyperparameter tuning trial. Defaults to False.

    Returns:
        result: The result of the training process.

    Logs:
        Information about setup, dataset loading, model wrapping, and training process.
    """

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])
    # Assuming `conf` is a dictionary containing your configuration
    trainer_config = conf.get('trainer', {})
    # Get KL loss coefficient with a default value of 10 if not present
    kl_loss_coefficient = trainer_config.get('kl_loss_coefficient', 10)
    # Get uncertainty with a default value of False if not present
    uncertainty = trainer_config.get('uncertainty', False)
    # Number of class labels present
    num_classes = conf["model"]["output_size"]

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # infer device id from rank

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    train_batch_size = conf['trainer']['train_batch_size']
    valid_batch_size = conf['trainer']['valid_batch_size']
    thread_workers = conf['trainer']['thread_workers']
    valid_thread_workers = conf['trainer']['valid_thread_workers'] if 'valid_thread_workers' in conf['trainer'] else thread_workers

    # load dataset and sampler

    train_dataset, train_sampler, data = load_dataset_and_sampler(conf, world_size, rank, split="train", return_df=True)
    valid_dataset, valid_sampler = load_dataset_and_sampler(conf, world_size, rank, split="val")
    test_dataset, test_sampler = load_dataset_and_sampler(conf, world_size, rank, split="test")

    # setup the dataloder for this process

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=valid_thread_workers,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=valid_thread_workers,
        drop_last=False
    )

    # model

    m = CategoricalDNN(**conf["model"])

    # have to send the module to the correct device first

    m.to(device)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, m, device)

    # Load model weights (if any), an optimizer, scheduler, and gradient scaler

    model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    # Train and validation losses

    if uncertainty:
        train_criterion = EDLDigammaLoss(num_classes, kl_loss_coefficient)
        valid_criterion = EDLDigammaLoss(num_classes, kl_loss_coefficient)
    else:
        train_criterion = torch.nn.CrossEntropyLoss()
        valid_criterion = torch.nn.CrossEntropyLoss()

    # Initialize a trainer object

    trainer = Trainer(
        model,
        rank,
        module=(conf["trainer"]["mode"] == "ddp"),
        uncertainty=uncertainty
    )

    # Fit the model
    classifier_metrics = MetricsCalculator(use_uncertainty=uncertainty)

    result = trainer.fit(
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        classifier_metrics,
        trial=trial
    )

    # Predict with the model
    logging.info("Predicting on the data splits with the trained model. This may take some time")

    # train loader needs reinitialized so we do not drop last batch
    # valid and test we did not do that since doesnt affect gradient
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=False
    )

    splits = ["train", "val", "test"]
    loaders = [train_loader, valid_loader, test_loader]
    all_results = defaultdict(dict)

    for split, dataloader in zip(splits, loaders):

        result = trainer.predict(
            conf,
            dataloader,
            valid_criterion,
            classifier_metrics,
            split
        )

        metrics = result["metrics"]
        mu = result["mu"]
        u = result["dempster-shafer"]
        aleatoric = result["aleatoric"]
        epistemic = result["epistemic"]
        total = result["total"]

        # Add predictions back to the DataFrame
        df = data[split].copy()

        df["u"] = u
        df[[f"y_pred_prob_{label}" for label in range(num_classes)]] = mu
        df[[f"aleatorc_{label}" for label in range(num_classes)]] = aleatoric
        df[[f"epistemic_{label}" for label in range(num_classes)]] = epistemic
        df[[f"total_{label}" for label in range(num_classes)]] = total

        for key, values in metrics.items():
            if isinstance(values, list):
                values = values[0]
            all_results[key][split] = values

        df.to_csv(os.path.join(conf['save_loc'], f"{split}.csv"))

    mets = pd.DataFrame.from_dict(all_results, orient='index')
    mets.to_csv(os.path.join(conf['save_loc'], "metrics.csv"))

    return result


class Objective(BaseObjective):
    """
    Objective class for hyperparameter tuning with Optuna.

    Args:
        config (dict): Configuration dictionary for the objective.
        metric (str, optional): The metric to optimize. Defaults to "val_loss".
        device (str, optional): The device to use. Defaults to "cpu".
    """

    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        """
        Train the model with the given trial configuration.

        Args:
            trial (optuna.Trial): The Optuna trial object for hyperparameter tuning.
            conf (dict): Configuration dictionary for the model and training.

        Returns:
            float: The result of the training process.

        Raises:
            optuna.TrialPruned: If the trial is to be pruned due to various reasons.
        """

        conf['trainer']['train_batch_size'] = conf['data']['batch_size']
        conf['trainer']['valid_batch_size'] = conf['data']['batch_size']
        conf['valid_loss']['coef'] = conf['train_loss']['coef']

        try:
            return main(0, 1, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "non-singleton" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to shape mismatch: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "infinity" in str(E) or "nan" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to encountering a nan in metrics calculations: {str(E)}."
                )
                raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


if __name__ == "__main__":

    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
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
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    # Update config using override options
    if mode and mode in ["none", "ddp", "fsdp"]:
        logging.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_pbs_jobs(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_distributed_jobs(config, script_path)
        sys.exit()

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
