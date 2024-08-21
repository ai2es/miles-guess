import warnings
import os
import sys
import yaml
import wandb
import shutil
import logging
import pandas as pd
import importlib

from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from mlguess.torch.distributed import distributed_model_wrapper
from mlguess.pbs import launch_pbs_jobs, launch_distributed_jobs
from mlguess.torch.checkpoint import load_model_state
from mlguess.torch.trainer_regression import Trainer
from mlguess.torch.regression_losses import LipschitzMSELoss
from mlguess.torch.models import seed_everything, DNN
from mlguess.regression_metrics import regression_metrics

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


def load_dataset_and_sampler(conf, world_size, rank, split, seed=42):
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

    # Use the function to import your script
    torch_dataset = import_class_from_path(conf["data"]["dataset_name"], conf["data"]["dataset_path"])
    dataset = torch_dataset(conf, split=split)

    # Pytorch sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False,
        drop_last=False
    )
    logging.info(f"Loaded a {split} torch dataset, and a distributed sampler")

    return dataset, sampler


def main(rank, world_size, conf, trial=False):

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])

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

    train_dataset, train_sampler = load_dataset_and_sampler(conf, world_size, rank, "train")
    valid_dataset, valid_sampler = load_dataset_and_sampler(conf, world_size, rank, "valid")
    test_dataset, test_sampler = load_dataset_and_sampler(conf, world_size, rank, "test")

    # setup the dataloder for this process

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=False
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
        pin_memory=False,
        num_workers=valid_thread_workers,
        drop_last=False
    )

    # model

    m = DNN(**conf["model"])

    # add the variance computed in the dataset to the model

    m.training_var = train_dataset.training_var

    # have to send the module to the correct device first

    m.to(device)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, m, device)

    # Load model weights

    m = load_model_state(conf, m, device)

    # Evaluation loss

    valid_criterion = LipschitzMSELoss(**conf["valid_loss"])

    # Initialize a trainer object

    trainer = Trainer(model, rank, module=(conf["trainer"]["mode"] == "ddp"))

    # Predict with the model
    splits = ["train", "valid", "test"]
    loaders = [train_loader, valid_loader, test_loader]
    all_results = defaultdict(dict)

    for split, dataloader in zip(splits, loaders):

        result = trainer.predict(
            conf,
            dataloader,
            valid_criterion,
            regression_metrics,
            train_dataset.y_scaler,
            split
        )

        metrics = result["metrics"]
        mu = result["mu"]
        aleatoric = result["aleatoric"]
        epistemic = result["epistemic"]
        total = result["total"]

        # Add predictions back to the DataFrame
        if split == "train":
            df = train_loader.dataset.train_data
        elif split == "valid":
            df = valid_loader.dataset.valid_data
        elif split == "test":
            df = test_loader.dataset.test_data

        df["mu"] = mu
        df["aleatoric"] = aleatoric
        df["epistemic"] = epistemic
        df["total_uncertainty"] = total

        for key, values in metrics.items():
            if isinstance(values, list):
                values = values[0]
            all_results[key][split] = values

        df.to_csv(os.path.join(conf['save_loc'], f"{split}.csv"))

    mets = pd.DataFrame.from_dict(all_results, orient='index')
    mets.to_csv(os.path.join(conf['save_loc'], "metrics.csv"))

    return result


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
        "-w",
        "--wandb",
        dest="wandb",
        type=int,
        default=0,
        help="Use wandb. Default = False"
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
    use_wandb = int(args_dict.pop("wandb"))
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

    if use_wandb:  # this needs updated
        wandb.init(
            # set the wandb project where this run will be logged
            project="Derecho parallelism",
            name=f"Worker {os.environ['RANK']} {os.environ['WORLD_SIZE']}",
            # track hyperparameters and run metadata
            config=conf
        )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
