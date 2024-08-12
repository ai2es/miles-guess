import yaml
import copy
import torch
import numpy as np
from mlguess.preprocessing import load_preprocessing
from mlguess.keras.data import load_ptype_uq, preprocess_data
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, conf, split='train',  data_split=0):
        input_features = []
        for features in ["TEMP_C", "T_DEWPOINT_C", "UGRD_m/s", "VGRD_m/s"]:
            input_features += conf["data"][features]
        output_features = conf["data"]["ptypes"]

        # Load data
        _conf = copy.deepcopy(conf)
        _conf.update(conf["data"])
        data = load_ptype_uq(_conf, data_split=data_split, verbose=1, drop_mixed=False)
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
            groups=groups,
        )

        self.scalers = scalers
        self.inputs = torch.tensor(scaled_data[f"{split}_x"][input_features].values, dtype=torch.float32)
        self.targets = torch.tensor(scaled_data[f"{split}_y"], dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y


if __name__ == "__main__":

    config_file = "../../config/evidential_classifier_torch.yml"

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    batch_size = 32

    # Create dataset instances
    train_dataset = CustomDataset(conf, split='train')
    valid_dataset = CustomDataset(conf, split='val')
    test_dataset = CustomDataset(conf, split='test')

    x, y = train_dataset.__getitem__(0)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
