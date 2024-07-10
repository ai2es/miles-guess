import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from mlguess.preprocessing import load_preprocessing
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, conf, split='train'):
        config = conf["data"]
        data_path = config['data_path']
        input_cols = config['input_cols']
        output_cols = config['output_cols']
        flat_seed = config['split_params']['flat_seed']
        data_seed = config['split_params']['data_seed']
        train_size = config['split_ratios']['train_size']
        valid_size = config['split_ratios']['valid_size']

        # Load data
        data = pd.read_csv(data_path)
        data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])
        data["year"] = data["Time"].apply(lambda x: str(x).split("-")[0])

        # Split data into train and test
        gsp = GroupShuffleSplit(n_splits=1, random_state=flat_seed, train_size=train_size)
        splits = list(gsp.split(data, groups=data["year"]))
        train_index, test_index = splits[0]
        train_data, self.test_data = data.iloc[train_index].copy(), data.iloc[test_index].copy()

        # Split train data into train and validation
        gsp = GroupShuffleSplit(n_splits=1, random_state=flat_seed, train_size=valid_size)
        splits = list(gsp.split(train_data, groups=train_data["year"]))
        train_index, valid_index = splits[data_seed]
        self.train_data, self.valid_data = train_data.iloc[train_index].copy(), train_data.iloc[valid_index].copy()

        # Initialize scalers
        self.x_scaler, self.y_scaler = load_preprocessing(conf, seed=conf["seed"])

        # Fit scalers on training data
        self.x_scaler.fit(train_data[input_cols])
        self.y_scaler.fit(train_data[output_cols])

        # Compute var on the total training data set
        self.training_var = [
            np.var(train_data[output_cols].values)#np.var(self.y_scaler.transform(train_data[output_cols]))
            for i in range(self.train_data[output_cols].shape[-1])
        ]

        # Transform the individual splits
        if split == 'train':
            self.inputs = self.x_scaler.transform(self.train_data[input_cols])
            self.targets = self.y_scaler.transform(self.train_data[output_cols])
        elif split == 'valid':
            self.inputs = self.x_scaler.transform(self.valid_data[input_cols])
            self.targets = self.y_scaler.transform(self.valid_data[output_cols])
        elif split == 'test':
            self.inputs = self.x_scaler.transform(self.test_data[input_cols])
            self.targets = self.y_scaler.transform(self.test_data[output_cols])
        else:
            raise ValueError("Invalid split value. Choose from 'train', 'valid', 'test'.")

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y


if __name__ == "__main__":
    # Example configuration dictionary
    config = {
        "data": {
            "data_path": "/glade/p/cisl/aiml/ai2es/surfacelayer/cabauw_derived_data_20210720.csv",
            "input_cols": [
                'wind_speed:10_m:m_s-1',
                'potential_temperature_skin_change:10_m:K_m-1',
                'bulk_richardson:10_m:None',
                'mixing_ratio_skin_change:2_m:g_kg-1_m-1'
            ],
            "output_cols": ['friction_velocity:surface:m_s-1'],
            "split_params": {
                "flat_seed": 42,
                "data_seed": 0
            },
            "split_ratios": {
                "train_size": 0.9,
                "valid_size": 0.885
            },
            "scaler_x": {
                "params": {
                    "copy": True,
                    "with_mean": True,
                    "with_std": True
                },
                "type": "quantile"
            },
            "scaler_y": {
                "params": {
                    "copy": True,
                    "with_mean": True,
                    "with_std": True
                },
                "type": "normalize"
            },
            "batch_size": 9163,
            "data_path": '/glade/p/cisl/aiml/ai2es/surfacelayer/cabauw_derived_data_20210720.csv'
        },
        "seed": 42
    }

    # Create dataset instances
    train_dataset = CustomDataset(config, split='train')
    valid_dataset = CustomDataset(config, split='valid')
    test_dataset = CustomDataset(config, split='test')

    x, y = train_dataset.__getitem__(0)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

    # Print dataset sizes
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(valid_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # Example: Iterate through the train data loader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Targets: {targets}')
        if batch_idx == 1:  # Limit to 2 batches for demonstration
            break
