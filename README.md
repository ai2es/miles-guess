# MILES-Guess
Generalized Uncertainty for Earth System Science (GUESS)

Developed by the Machine Ingetration and Learning for Earth Systems (MILES) group at the NSF National Center for Atmospheric Research (NCAR), Boulder CO, USA

## Contributors 
* John Schreck
* David John Gagne
* Charlie Becker
* Gabrielle Gantos
* Dhamma Kimpara
* Thomas Martin

## Documentation
Full documentation is [here](https://miles-guess.readthedocs.io/en/latest/).

## Quick Setup

Install in your Python environment with the following command:
```bash
pip install miles-guess
```
If you want to install a particular backend (tensorflow, tensorflow_gpu, torch, jax): 
```bash
pip install miles-guess[<backend>]
```
## Setup from Scratch

Install the Miniconda Python installer available
[here](https://docs.conda.io/en/latest/miniconda.html).

First clone the miles-guess repo from github.
```bash
git clone https://github.com/ai2es/miles-guess.git`
cd miles-guess
```

Create a conda environment for non-Casper/Derecho users:
```bash
mamba env create -f environment.yml`
conda activate guess`
```

Create a conda environment for Casper/Derecho users including Tensorflow 2.15 with GPU support.
```bash
mamba env create -f environment_gpu.yml`
conda activate guess
```

## Using miles-guess

The law of total variance for each model prediction target may be computed as

$$LoTV = E[\sigma^2] + Var[\mu]$$ 

which is the sum of aleatoric and epistemic contributions, respectively. The MILES-GUESS package contains options for using either Keras or PyTorch for computing quantites according to the LoTV as well as utilizing Dempster-Shafer theory uncertainty in the classifier case. 

For detailed information about training with Keras, refer to [the Keras training details README](docs/source/keras.md). There three scripts for training three regression models, and one for training categorical models. The regression examples are trained on our surface layer ("SL") dataset for predicting latent heat and other quantities, 
and the categorical example is trained on a precipitation dataset ("p-type").

For pyTorch, please visit the [the pyTorch training details README](docs/source/torch.md) where details on training scripts for both evidential standard classification tasks are detailed. Torch examples use the same datasets as the Keras models. The torch training code will also scale on GPUs, and is compatitible with DDP and FSDP.

<!--
### 1a. Train/evaluate a deterministic multi-layer perceptrion (MLP) on the SL dataset:
```bash
python3 applications/train_mlp_SL.py -c config/model_mlp_SL.yml
```

### 1b. Train/evaluate a parametric "Gaussian" MLP on the SL dataset:
```bash

python applications/train_gaussian_SL.py -c config/model_gaussian_SL.yml
```

### 1c. Train/evaluate a parametric "normal-inverse gamma" (evidential) MLP on the SL dataset:
```bash
python applications/train_evidential_SL.py -c config/model_evidential_SL.yml
```

### 2a. Train a categorical MLP classifier on the p-type dataset:
```bash
python applications/train_classifier_ptype.py -c config/model_classifier_ptype.yml
```

### 2b. Train an evidential MLP classifier on the p-type dataset:
```bash
python applications/train_classifier_ptype.py -c config/model_evidential_ptype.yml
```

### 2c. Evaluate a categorical/evidential classifier on the p-type dataset:
```bash
python applications/evaluate_ptype.py -c config/model_classifier_ptype.yml
```


## 3. Ensembling modes for the deterministic model (1a)

There are four "modes" for training the deterministic MLP (1a) that are controlled using the "ensemble" field in a model configuration.
```yaml
ensemble:
    n_models: 100
    n_splits: 20
    monte_carlo_passes: 0
```
where n_models means the number of models to train using a fixed data split with variable initial weight initializations, n_splits means the number of models to train using variable training and validation splits (random initializations), and mc_carlo_passes means the number of MC-dropout evaluations performed on a given input to the model. 

### 3a. Single Mode
```yaml
ensemble:
    n_models: 1
    n_splits: 1
    monte_carlo_passes: 0
```
Train a single deterministic model (no uncertainty evaluation). If MC passes > 0, an ensemble is created after the model finishes training.

### 3b. Data Mode
```yaml
ensemble:
    n_models: 1
    n_splits: 10
    monte_carlo_passes: 100
```
Create an ensemble of models (random initialization) using cross-validation splits. If MC passes > 0, an ensemble is created after each model finishes training on the test holdout. The LOTV may then be applied to the ensemble created from cross-validation. Otherwise a single ensemble is created but the LOTV is not applied. 

### 3c. Model Mode
```yaml
ensemble:
    n_models: 10
    n_splits: 1
    monte_carlo_passes: 100
```
Create an ensemble of models using a fixed train/validation/test data split and variable model layer weight initializations. If MC passes > 0, an ensemble is created after each model finishes training. The LOTV may then be applied to the ensemble created from variable weight initializations to obtain uncertainty estimations for each prediction target. Otherwise a single ensemble is created but the LOTV is not applied. 

### 3d. Ensemble Mode
```yaml
ensemble:
    n_models: 1
    n_splits: 1
    monte_carlo_passes: 0
```
Create an ensemble of ensembles. The first ensemble is created using cross validation and a fixed weight initialization, from which a mean and variance may be obtained for each prediction target. The second ensemble is created by varying the weight initalization that can then be used with the LOTV to obtain uncertainty estimations for each prediction target. The MC steps field is ignored in ensemble mode. 

## 4. Ensembling modes for the Gaussian parametric model (1b)
There are three "modes" for training the Gaussian MLP (1b).

### 4a. Single Mode
```yaml
ensemble:
    n_models: 1
    n_splits: 1
    monte_carlo_passes: 0
```
Train a single deterministic model (no LOTV evaluation). If MC passes > 0, an ensemble is created after the model finishes training (LOTV evaluation).

### 4b. Data Mode
```yaml
ensemble:
    n_models: 1
    n_splits: 10
    monte_carlo_passes: 0
```
Create an ensemble of models using cross-validation splits, and then LOTV evaluation.

### 4c. Model Mode
```yaml
ensemble:
    n_models: 10
    n_splits: 1
    monte_carlo_passes: 0
```
Create an ensemble of models using different random initializations and a fixed cross-validation split, and then LOTV evaluation.

## Configuration files
In addition to the ensemble field, the other fields in the configuration file are 

```yaml
seed: 1000
save_loc: "/path/to/save/directory"
training_metric: "val_mae"
direction: "min"
```

where seed allows for reproducability, save_loc is where data will be saved, and training metric and direction are used as the validation metric (and direction).

For regression tasks, other fields in a configuration file are model and callbacks:
```yaml

model:
    activation: relu
    batch_size: 193
    dropout_alpha: 0.2
    epochs: 200
    evidential_coef: 0.6654439861214466
    hidden_layers: 3
    hidden_neurons: 6088
    kernel_reg: l2
    l1_weight: 0.0
    l2_weight: 7.908676527243475e-10
    lr: 3.5779279071474884e-05
    metrics: mae
    optimizer: adam
    uncertainties: true
    use_dropout: true
    use_noise: false
    verbose: 2
      
callbacks:
  EarlyStopping:
    monitor: "val_loss"
    patience: 5
    mode: "min"
    verbose: 0
  ReduceLROnPlateau: 
    monitor: "val_loss"
    factor: 0.1
    patience: 2
    min_lr: 1.0e-12
    min_delta: 1.0e-08
    mode: "min"
    verbose: 0
  CSVLogger:
    filename: "training_log.csv"
    separator: ","
    append: False
  ModelCheckpoint:
    filepath: "model.h5"
    monitor: "val_loss"
    save_weights: True
    save_best_only: True
    mode: "min"
    verbose: 0
```

For categorical tasks, the model field changes slightly:

```yaml
model:
    activation: leaky
    balanced_classes: 1
    batch_size: 3097
    dropout_alpha: 0.31256692323263807
    epochs: 200
    hidden_layers: 4
    hidden_neurons: 6024
    loss: categorical_crossentropy
    loss_weights:
    - 21.465788717561477
    - 83.31367732936326
    - 136.50944842077058
    - 152.62042204485107
    lr: 0.0004035503144482269
    optimizer: adam
    output_activation: softmax
    use_dropout: 1
    verbose: 0
```
where the user has two options:

(1) A standard deterministic classifier is trained when 
```yaml 
    loss: cateogical_crossentropy
    output_activation: softmax
```
(2) An evidential classifier is trained when 
```yaml 
    loss: dirichlet
    output_activation: linear
```

Callbacks are not required in regression training, however a custom callback which tracks the current epoch is requried for the categorical model and is added automatically (the user does not need to specify it in the callbacks field). The user may add any other supported keras callback by adding the relevant fields to the callbacks field. 

Depending on the problem, a data field is customized and also present in the configuration files. See the examples for surface layer and p-type data sets for more details. 


## ECHO hyperparameter optimization 

Configuration files are also supplied for use with the Earth Computing Hyperparameter Optimization (ECHO) package. See the echo package https://github.com/NCAR/echo-opt/tree/main/echo for more details on the configuration fields. -->
