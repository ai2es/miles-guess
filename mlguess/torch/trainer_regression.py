import gc
import logging
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
import optuna
from mlguess.torch.checkpoint import TorchFSDPCheckpointIO


def cleanup():
    """Clean up and destroy the process group for distributed training.

    This function is used to release resources and finalize the distributed training environment
    by destroying the process group. It should be called at the end of distributed training.

    Returns:
        None
    """
    dist.destroy_process_group()


def accum_log(log, new_logs):
    """Accumulate new log values into the existing log dictionary.

    Args:
        log (dict): The existing log dictionary to which new values will be added.
        new_logs (dict): Dictionary containing new log values to be accumulated.

    Returns:
        dict: Updated log dictionary with accumulated values.

    Example:
        old_log = {'loss': 1.0, 'accuracy': 0.8}
        new_log = {'loss': 0.5, 'accuracy': 0.9}
        updated_log = accum_log(old_log, new_log)
        # updated_log will be {'loss': 1.5, 'accuracy': 1.7}
    """
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class Trainer:
    def __init__(self, model, rank, module=False):
        """Initialize the Trainer class.

        Args:
            model (nn.Module): The model to be trained.
            rank (int): The rank of the current process (used for distributed training).
            module (bool): Whether the model is wrapped in a `DistributedDataParallel` module. Default is False.
        """
        super(Trainer, self).__init__()
        self.model = model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

        if module:
            self.model = self.model.module

    # Training function.
    def train_one_epoch(
        self,
        epoch,
        conf,
        trainloader,
        optimizer,
        criterion,
        scaler,
        scheduler,
        metrics,
        transform=None
    ):
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            conf (dict): Configuration dictionary containing training settings.
            trainloader (DataLoader): DataLoader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            criterion (callable): Loss function used for training.
            scaler (torch.cuda.amp.GradScaler): Scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            metrics (dict): Dictionary of metric functions to evaluate model performance.
            transform (optional, callable): Transformation function to apply to the data. Default is None.

        Returns:
            dict: Dictionary containing training metrics for the epoch.
        """
        batches_per_epoch = conf['trainer']['batches_per_epoch']
        grad_accum_every = conf['trainer']['grad_accum_every']
        amp = conf['trainer']['amp']
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "lambda":
            scheduler.step()

        # set up a custom tqdm
        batches_per_epoch = (
            batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)
        )

        batch_group_generator = tqdm.tqdm(
            enumerate(trainloader),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False
        )

        results_dict = defaultdict(list)

        self.model.train()
        for i, (x, y) in batch_group_generator:

            logs = {}

            commit_loss = 0.0

            with autocast(enabled=amp):
                x = x.to(self.device)
                y_pred = self.model(x)
                gamma, nu, alpha, beta = y_pred
                y = y.to(device=self.device, dtype=x.dtype)
                loss = criterion(gamma, nu, alpha, beta, y.to(x.dtype))

                # Metrics
                y_pred = (_.cpu().detach() for _ in y_pred)
                mu, ale, epi, total = self.model.predict_uncertainty(y_pred, y_scaler=transform)
                if transform:
                    y = transform.inverse_transform(y.cpu())
                else:
                    y = y.cpu()
                metrics_dict = metrics(y, mu.numpy(), total, split="train")
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[name].append(value[0].item())

                loss = loss.mean() + commit_loss

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {'loss': loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    raise optuna.TrialPruned()
                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_mae: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_mae"]),
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "cosine-annealing":
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(
        self,
        epoch,
        conf,
        valid_loader,
        criterion,
        metrics,
        transform=None
    ):
        """Validate the model on the validation dataset.

        Args:
            epoch (int): The current epoch number.
            conf (dict): Configuration dictionary containing validation settings.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            criterion (callable): Loss function used for validation.
            metrics (dict): Dictionary of metric functions to evaluate model performance.
            transform (optional, callable): Transformation function to apply to the data. Default is None.

        Returns:
            dict: Dictionary containing validation metrics for the epoch.
        """
        self.model.eval()

        valid_batches_per_epoch = conf['trainer']['valid_batches_per_epoch']
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        valid_batches_per_epoch = (
            valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)
        )

        batch_group_generator = tqdm.tqdm(
            enumerate(valid_loader),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False
        )

        self.model.eval()
        for i, (x, y) in batch_group_generator:

            with torch.no_grad():
                x = x.to(self.device)
                y_pred = self.model(x)
                gamma, nu, alpha, beta = y_pred
                y = y.to(device=self.device, dtype=x.dtype)
                loss = criterion(gamma, nu, alpha, beta, y)

                # Metrics
                y_pred = (_.cpu() for _ in y_pred)
                mu, ale, epi, total = self.model.predict_uncertainty(y_pred, y_scaler=transform)
                if transform:
                    y = transform.inverse_transform(y.cpu())
                metrics_dict = metrics(y, mu.numpy(), total, split="valid")

                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[name].append(value[0].item())

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if distributed:
                    torch.distributed.barrier()
                results_dict["valid_loss"].append(batch_loss[0].item())

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_mae"])
                )
                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

                if i >= valid_batches_per_epoch and i > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def predict(
        self,
        conf,
        test_loader,
        criterion,
        metrics,
        transform=None,
        split=None
    ):
        """Make predictions with the model on the test dataset.

        Args:
            conf (dict): Configuration dictionary containing prediction settings.
            test_loader (DataLoader): DataLoader for the test dataset.
            criterion (callable): Loss function used for evaluating predictions.
            metrics (dict): Dictionary of metric functions to evaluate model performance.
            transform (optional, callable): Transformation function to apply to the data. Default is None.
            split (optional, callable): Function to split the predictions if needed. Default is None.

        Returns:
            tuple: Tuple containing:
                - predictions (torch.Tensor): The model's predictions on the test dataset.
                - metrics (dict): Dictionary containing evaluation metrics.
        """
        self.model.eval()
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)
        mu_list, ale_list, epi_list, y_list = [], [], [], []

        batch_group_generator = tqdm.tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            leave=True,
            disable=True if self.rank > 0 else False
        )

        for i, (x, y) in batch_group_generator:
            with torch.no_grad():
                x = x.to(self.device)
                y_pred = self.model(x)
                gamma, nu, alpha, beta = y_pred
                y = y.to(device=self.device, dtype=x.dtype)
                loss = criterion(gamma, nu, alpha, beta, y)

                # Metrics
                y_pred = (_.cpu() for _ in y_pred)
                mu, ale, epi, total = self.model.predict_uncertainty(y_pred, y_scaler=transform)
                mu_list.append(mu)
                ale_list.append(ale)
                epi_list.append(epi)
                y_list.append(y.cpu())

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if distributed:
                    torch.distributed.barrier()
                results_dict[f"{split}_loss"].append(batch_loss[0].item())

                # Print to tqdm
                to_print = f'{split} loss: {np.mean(results_dict[f"{split}_loss"]):.6f}'
                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

        # Concatenate arrays
        mu = np.concatenate(mu_list, axis=0)
        ale = np.concatenate(ale_list, axis=0)
        epi = np.concatenate(epi_list, axis=0)
        total = ale + epi
        y = np.concatenate(y_list, axis=0)

        if transform:
            y = transform.inverse_transform(y)

        # Compute metrics
        metrics_dict = metrics(y, mu, total, split=split)
        for name, value in metrics_dict.items():
            value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
            if distributed:
                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
            results_dict[name].append(value[0].item())
        results_dict[f"{split}_loss"].append(np.mean(results_dict[f"{split}_loss"]))

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # Clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return {
            'metrics': results_dict,
            'mu': mu,
            'aleatoric': ale,
            'epistemic': epi,
            'total': total
        }

    def fit(
        self,
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        transform=None,
        trial=False
    ):
        """Train and validate the model.

        Args:
            conf (dict): Configuration dictionary containing training and validation settings.
            train_loader (DataLoader): DataLoader for the training dataset.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            train_criterion (callable): Loss function used for training.
            valid_criterion (callable): Loss function used for validation.
            scaler (torch.cuda.amp.GradScaler): Scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            metrics (dict): Dictionary of metric functions to evaluate model performance.
            transform (optional, callable): Transformation function to apply to the data. Default is None.
            trial (bool): Whether this is a trial run. Default is False.

        Returns:
            dict: Dictionary containing training and validation metrics.
        """
        save_loc = conf['save_loc']
        start_epoch = conf['trainer']['start_epoch']
        epochs = conf['trainer']['epochs']
        skip_validation = conf['trainer']['skip_validation'] if 'skip_validation' in conf['trainer'] else False

        # set up the training optimization metric used in callbacks
        if "training_metric" in conf["trainer"]:
            training_metric = conf["trainer"]["training_metric"]
        else:
            training_metric = "valid_loss"
        training_metric = "train_loss" if skip_validation else training_metric
        # logger.info()

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(save_loc, "training_log.csv"))
            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        for epoch in range(start_epoch, epochs):

            logging.info(f"Beginning epoch {epoch}")

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch,
                conf,
                train_loader,
                optimizer,
                train_criterion,
                scaler,
                scheduler,
                metrics,
                transform
            )

            ############
            #
            # Validation
            #
            ############

            if skip_validation:

                valid_results = train_results

            else:

                valid_results = self.predict(
                    conf,
                    valid_loader,
                    valid_criterion,
                    metrics,
                    transform,
                    split="valid"
                )["metrics"]

                # this version of validation computes metrics batch-by-batch, which may affect metrics computed through binning

                # valid_results = self.validate(
                #     epoch,
                #     conf,
                #     valid_loader,
                #     valid_criterion,
                #     metrics,
                #     transform
                # )

            #################
            #
            # Save results
            #
            #################

            # update the learning rate if epoch-by-epoch updates
            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "plateau":
                scheduler.step(np.mean(valid_results["valid_loss"]))

            # Put things into a results dictionary -> dataframe

            results_dict["epoch"].append(epoch)
            all_keys = sorted(["_".join(k.split("_")[1:]) for k in train_results.keys()])
            for name in all_keys:  # "acc", "mae"
                results_dict[f"train_{name}"].append(np.mean(train_results[f"train_{name}"]))
                results_dict[f"valid_{name}"].append(np.mean(valid_results[f"valid_{name}"]))
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            df = pd.DataFrame.from_dict(results_dict).reset_index()

            # Save the dataframe to disk

            if trial:
                df.to_csv(
                    os.path.join(f"{save_loc}", "trial_results", f"training_log_{trial.number}.csv"),
                    index=False,
                )
            else:
                df.to_csv(os.path.join(f"{save_loc}", "training_log.csv"), index=False)

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:

                if conf["trainer"]["mode"] != "fsdp":

                    if self.rank == 0:

                        # Save the current model

                        logging.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                            'scaler_state_dict': scaler.state_dict()
                        }
                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                else:

                    logging.info(f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                    # Initialize the checkpoint I/O handler

                    checkpoint_io = TorchFSDPCheckpointIO()

                    # Save model and optimizer checkpoints

                    checkpoint_io.save_unsharded_model(
                        self.model,
                        os.path.join(save_loc, "model_checkpoint.pt"),
                        gather_dtensor=True,
                        use_safetensors=False,
                        rank=self.rank
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pt"),
                        gather_dtensor=True,
                        rank=self.rank
                    )

                    # Still need to save the scheduler and scaler states, just in another file for FSDP

                    state_dict = {
                        "epoch": epoch,
                        'scheduler_state_dict': scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                        'scaler_state_dict': scaler.state_dict()
                    }

                    torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

                # save if this is the best model seen so far
                if (self.rank == 0) and (results_dict[training_metric][-1] == min(results_dict[training_metric])):
                    if conf["trainer"]["mode"] == "fsdp":
                        pass
                    else:
                        shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            # Report result to the trial
            # if trial:
            #     trial.report(results_dict[training_metric][-1], step=epoch)

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict[training_metric])
                if j == min(results_dict[training_metric])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf['trainer']['stopping_patience']:
                logging.info("Stopping early")
                break

            # Stop training if we get too close to the wall time
            if 'stop_after_epoch' in conf['trainer']:
                if conf['trainer']['stop_after_epoch']:
                    break

        best_epoch = [
            i for i, j in enumerate(results_dict[training_metric]) if j == min(results_dict[training_metric])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
