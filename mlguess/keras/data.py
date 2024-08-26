import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    RobustScaler,
    QuantileTransformer,
)

from bridgescaler.group import GroupMinMaxScaler, GroupRobustScaler, GroupStandardScaler
from sklearn.model_selection import GroupShuffleSplit


logger = logging.getLogger(__name__)


def load_ptype_uq(conf, data_split=0, verbose=0, drop_mixed=False):

    # Load
    df = pd.read_parquet(conf["data_path"])

    # Drop mixed cases
    if drop_mixed:
        logger.info("Dropping data points with mixed observations")
        c1 = df["ra_percent"] == 1.0
        c2 = df["sn_percent"] == 1.0
        c3 = df["pl_percent"] == 1.0
        c4 = df["fzra_percent"] == 1.0
        condition = c1 | c2 | c3 | c4
        df = df[condition].copy()

    # QC-Filter
    qc_value = str(conf["qc"])
    cond1 = df[f"wetbulb{qc_value}_filter"] == 0.0
    cond2 = df["usa"] == 1.0
    dg = df[cond1 & cond2].copy()

    dg["day"] = dg["datetime"].apply(lambda x: str(x).split(" ")[0])
    dg["id"] = range(dg.shape[0])

    # Select test cases
    test_days_c1 = dg["day"].isin(
        [day for case in conf["case_studies"].values() for day in case]
    )
    test_days_c2 = dg["day"] >= conf["test_cutoff"]
    test_condition = test_days_c1 | test_days_c2

    # Partition the data into trainable-only and test-only splits
    train_data = dg[~test_condition].copy()
    test_data = dg[test_condition].copy()

    # Make N train-valid splits using day as grouping variable, return "data_split" split
    gsp = GroupShuffleSplit(
        n_splits=conf["ensemble"]["n_splits"],
        random_state=conf["seed"],
        train_size=conf["train_size1"],
    )
    splits = list(gsp.split(train_data, groups=train_data["day"]))

    train_index, valid_index = splits[data_split]
    train_data, valid_data = (
        train_data.iloc[train_index].copy(),
        train_data.iloc[valid_index].copy(),
    )

    size = df.shape[0]
    logger.info("Train, validation, and test fractions:")
    logger.info(
        f"{train_data.shape[0]/size}, {valid_data.shape[0]/size}, {test_data.shape[0]/size}"
    )
    data = {"train": train_data, "val": valid_data, "test": test_data}

    return data


def preprocess_data(
    data,
    input_features,
    output_features,
    scaler_type="standard",
    encoder_type="onehot",
    groups=[],
    seed=1000,
):
    """Function to select features and scale data for ML
    Args:
        data (dictionary of dataframes for training and validation data):
        input_features (list): Input features
        output_feature (list): Output feature
        scaler_type: Type of scaling to perform (supports "standard" and "minmax")
        encoder_type: Type of encoder to perform (supports "label" and "onehot")

    Returns:
        Dictionary of scaled and one-hot encoded data, dictionary of scaler objects
    """
    groupby = len(groups)

    scalar_obs = {
        "normalize": MinMaxScaler() if not groupby else GroupMinMaxScaler(),
        "symmetric": MinMaxScaler((-1, 1))
        if not groupby
        else GroupMinMaxScaler(),
        "standard": StandardScaler() if not groupby else GroupStandardScaler(),
        "robust": RobustScaler() if not groupby else GroupRobustScaler(),
        "quantile": QuantileTransformer(
            n_quantiles=1000, random_state=seed, output_distribution="normal"
        ),
        "quantile-uniform": QuantileTransformer(
            n_quantiles=1000, random_state=seed, output_distribution="uniform"
        ),
    }
    scalers, scaled_data = {}, {}
    scalers["input"] = scalar_obs[scaler_type]
    scalers["output_label"] = LabelEncoder()
    if encoder_type == "onehot":
        scalers["output_onehot"] = OneHotEncoder(sparse_output=False)

    if groupby and "quantile" not in scaler_type:
        scaled_data["train_x"] = pd.DataFrame(
            scalers["input"].fit_transform(
                data["train"][input_features], groups=groups
            ),
            columns=input_features,
        )
    else:
        scaled_data["train_x"] = pd.DataFrame(
            scalers["input"].fit_transform(data["train"][input_features]),
            columns=input_features,
        )
    scaled_data["val_x"] = pd.DataFrame(
        scalers["input"].transform(data["val"][input_features]), columns=input_features
    )
    scaled_data["test_x"] = pd.DataFrame(
        scalers["input"].transform(data["test"][input_features]), columns=input_features
    )
    if "left_overs" in data:
        scaled_data["left_overs_x"] = pd.DataFrame(
            scalers["input"].transform(data["left_overs"][input_features]),
            columns=input_features,
        )

    scalers["output_label"] = LabelEncoder()
    scaled_data["train_y"] = scalers["output_label"].fit_transform(
        np.argmax(data["train"][output_features].to_numpy(), 1)
    )
    scaled_data["val_y"] = scalers["output_label"].transform(
        np.argmax(data["val"][output_features].to_numpy(), 1)
    )
    scaled_data["test_y"] = scalers["output_label"].transform(
        np.argmax(data["test"][output_features].to_numpy(), 1)
    )
    if "left_overs" in data:
        scaled_data["left_overs_y"] = scalers["output_label"].transform(
            np.argmax(data["left_overs"][output_features].to_numpy(), 1)
        )

    if encoder_type == "onehot":
        scalers["output_onehot"] = OneHotEncoder(sparse_output=False)
        scaled_data["train_y"] = scalers["output_onehot"].fit_transform(
            np.expand_dims(scaled_data["train_y"], 1)
        )
        scaled_data["val_y"] = scalers["output_onehot"].transform(
            np.expand_dims(scaled_data["val_y"], 1)
        )
        scaled_data["test_y"] = scalers["output_onehot"].transform(
            np.expand_dims(scaled_data["test_y"], 1)
        )
        if "left_overs" in data:
            scaled_data["left_overs_y"] = scalers["output_onehot"].transform(
                np.expand_dims(scaled_data["left_overs_y"], 1)
            )

    return scaled_data, scalers
