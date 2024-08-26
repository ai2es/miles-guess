
import torch
import os

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)
from mlguess.torch.checkpoint import (
    TorchFSDPModel
)
from torch.nn.parallel import DistributedDataParallel as DDP
from mlguess.torch.mixed_precision import parse_dtype
import functools
import logging


def distributed_model_wrapper(conf, neural_network, device):
    """Wraps a neural network model in a distributed training wrapper (FSDP or DDP) based on configuration.

    Args:
        conf (dict): Configuration dictionary specifying the training setup, including the model type,
                     and training options such as mixed precision, CPU offloading, and activation checkpointing.
        neural_network (torch.nn.Module): The neural network model to be wrapped.
        device (torch.device): The device to which the model will be moved, usually a CUDA device.

    Returns:
        torch.nn.Module: The distributed model wrapped according to the configuration.
    """
    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])

    # FSDP polices
    if conf["trainer"]["mode"] == "fsdp":

        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000
        )

        # Mixed precision

        use_mixed_precision = conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False

        logging.info(f"Using mixed_precision: {use_mixed_precision}")

        if use_mixed_precision:
            for key, val in conf["trainer"]["mixed_precision"].items():
                conf["trainer"]["mixed_precision"][key] = parse_dtype(val)
            mixed_precision_policy = MixedPrecision(**conf["trainer"]["mixed_precision"])
        else:
            mixed_precision_policy = None

        # CPU offloading

        cpu_offload = conf["trainer"]["cpu_offload"] if "cpu_offload" in conf["trainer"] else False

        logging.info(f"Using CPU offloading: {cpu_offload}")

        # FSDP module

        model = TorchFSDPModel(
            neural_network,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload)
        )

        # activation checkpointing on the transformer blocks

        activation_checkpoint = conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False

        logging.info(f"Activation checkpointing: {activation_checkpoint}")

        if activation_checkpoint:

            # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            check_fn = lambda submodule: any(isinstance(submodule, cls) for cls in auto_wrap_policy)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn
            )

        # attempting to get around the launch issue we are having
        torch.distributed.barrier()

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(neural_network, device_ids=[device])
    else:
        model = neural_network

    return model
