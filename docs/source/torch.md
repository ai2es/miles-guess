# Pytorch in MILES-GUESS

Welcome to the pyTorch users page. The instructions below outline how to compute various UQ quantities like aleatoric and epistemic using different modeling approaches. Email schreck@ucar.edu for questions/concerns/fixes/etc

## Regression usage

There are two provided scripts which are mostly similar, one for training (and predicting), and one for loading a trained model and predicting. The second script serves as an example on how to load the model from a checkpoint as well as potentially scale inference across GPUs. You may only need to run the trainer script and not necessarily the predict script, as they will both save model predictions to file as well as metrics.

Run the training script with: `python applications/train_regressor_torch.py -c <path-to-config-file> [-l] [-m <mode>]`

Arguments:
- `-c, --config`: Path to the YAML configuration file (required)
- `-l`: Submit workers to PBS (optional, default: 0)
- `-m, --mode`: Set the training mode to 'none', 'ddp', or 'fsdp' (optional)

Example: 
`python trainer.py -c config.yml -m none -l 1`

Running distribued mode:

`torchrun [options] trainer.py -c config.yml -m ddp`

`torchrun [options] trainer.py -c config.yml -m fsdp`

The YAML configuration file should contain settings for the model, training, data, and pbs or slurm settings. For distributed training, set the `mode` in the config file or use the `-m` argument to specify 'ddp' or 'fsdp'. Use the `-l` flag to submit jobs to PBS or manually set up your distributed environment. If you plan to use more than 1 node, you may need to customize the torchrun for your system. FSDP is relatively hard to set up automatically, you will need to choose the model/data sharding policy on your own.

For more detailed information about configuration options and advanced usage, please refer to the code documentation and comments within the script.

[Optional]
Once a model is trained, if you would like to load the model after-the-fact and predict on the training splits, run

`python applications/predict_regressor_torch.py -c <path-to-config-file> [-l] [-m <mode>]`

which will load the trained model from disk and predict on the training splits and save them along with some computed metrics to disk. The predicted quanties include the task(s) predictions along with aleatoric and epistemic quantities.

## Classifier usage

For the classifier models, training and evaluating an evidential model on a dataset is performed in the same script, with options for distributed training using either DDP or FSDP. See the regression prediction script for an example on model checkpoint reloading and prediction.

Run the combined script with: 

`python applications/train_classifier_torch.py -c <path-to-config-file> [-l] [-m <mode>]`

Example: 

`python applications/train_classifier_torch.py -c config.yml -m none`

As noted this script will doubly train a model and then predict on the supplied training splits. The predicted quanties include the task(s) predictions along with the Dempster-Shafer uncertainty, and aleatoric and epistemic quantities for a $K$-class problem. Please see the full documentation for more. 

## Model and training configuration yaml

The most important fields for training evidential models with options are the trainer and model fields in the example config file. These fields apply and work with both classifier and regression models. 
```yaml
trainer:
    mode: fsdp # none, ddp, fsdp
    training_metric: "valid_loss"
    train_batch_size: *batch_size
    valid_batch_size: *batch_size
    batches_per_epoch: 500 # Set to 0 to use len(dataloader)
    valid_batches_per_epoch: 0
    learning_rate: 0.0015285262808755972
    weight_decay: 0.0009378550509012784
    start_epoch: 0
    epochs: 100
    amp: False
    grad_accum_every: 1
    grad_max_norm: 1.0
    thread_workers: 4
    valid_thread_workers: 4
    stopping_patience: 5
    load_weights: False
    load_optimizer: False
    use_scheduler: True
    # scheduler: {'scheduler_type': 'cosine-annealing', first_cycle_steps: 500, cycle_mult: 6.0, max_lr: 5.0e-04, min_lr: 5.0e-07, warmup_steps: 499, gamma: 0.7}
    scheduler: {scheduler_type: plateau, mode: min, factor: 0.1, patience: 2, cooldown: 2, min_lr: 1.0e-07, verbose: true, threshold: 1.0e-04}
    # scheduler: {'scheduler_type': 'lambda'}
  
model:
  input_size: *input_cols  # Reference to data:input_cols
  output_size: *output_cols  # Reference to data:output_cols
  layer_size: [1057, 1057, 1057, 1057, 1057]  # Example block sizes
  dr: [0.263, 0.263, 0.263, 0.263, 0.263]  # Dropout rates
  batch_norm: False  # Whether to use batch normalization
  lng: True  # Use the evidential layer (True) or not (False)
```

## Trainer Configuration

### General Settings

* **Mode:** Specifies the distributed training strategy.
    * Options: `none`, `ddp`, `fsdp`.
* **Training Metric:** Metric used for monitoring during training.
    * Example: `"valid_loss"`.
* **Batch Sizes:**
    * `train_batch_size`: Batch size used for training.
    * `valid_batch_size`: Batch size used for validation.
* **Epoch Configuration:**
    * `batches_per_epoch`: Number of batches per epoch during training. Set to `0` to use the entire dataloader length.
    * `valid_batches_per_epoch`: Number of batches per epoch during validation.
    * `start_epoch`: The epoch from which training starts.
    * `epochs`: Total number of epochs for training.
* **Learning Parameters:**
    * `learning_rate`: Initial learning rate.
    * `weight_decay`: Weight decay for regularization.
    * `amp`: Use Automatic Mixed Precision (AMP) if set to `True`.
    * `grad_accum_every`: Gradient accumulation steps.
    * `grad_max_norm`: Maximum norm for gradient clipping.
* **Multi-threading**:
    * `thread_workers`: Number of worker threads for training.
    * `valid_thread_workers`: Number of worker threads for validation.
* **Early Stopping**:
    * `stopping_patience`: Number of epochs with no improvement after which training will stop.
* **Checkpointing**:
    * `load_weights`: Load weights from a pre-trained model if `True`.
    * `load_optimizer`: Load optimizer state from a checkpoint if `True`.
* **Learning Rate Scheduler**:
    * `use_scheduler`: Apply learning rate scheduling if `True`.
    * `scheduler`: Dictionary containing scheduler configuration.

```yaml
# Example: Cosine Annealing Scheduler
scheduler:
  scheduler_type: cosine-annealing
  first_cycle_steps: 500
  cycle_mult: 6.0
  max_lr: 5.0e-04
  min_lr: 5.0e-07
  warmup_steps: 499
  gamma: 0.7

# Example: Plateau Scheduler
scheduler:
  scheduler_type: plateau
  mode: min
  factor: 0.1
  patience: 2
  cooldown: 2
  min_lr: 1.0e-07
  verbose: true
  threshold: 1.0e-04
```

## Model Configuration

### Input/Output Sizes

* `input_size`: Number of input features (referenced from data).
* `output_size`: Number of output features (referenced from data).

### Architecture

* `layer_size`: List defining the number of neurons in each layer.
* `dr`: List defining the dropout rates for each layer.
* `batch_norm`: Enable/Disable batch normalization. Set to `True` for enabling.

### Evidential Layer

* `lng`: Use evidential layer if `True`. Useful for uncertainty quantification.

### Classifier Models

* `output_activation`: Set to `softmax` for standard classification. If not set, the model will use evidential classification.
