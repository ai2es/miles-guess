from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import logging


logger = logging.getLogger(__name__)


def load_preprocessor(stype, params, seed=1000):
    # ignoring the **params argument for now for ease of using ECHO
    if stype == "standard":
        return StandardScaler()
    elif stype == "normalize":
        return MinMaxScaler((0, 1))
    elif stype == "symmetric":
        return MinMaxScaler((-1, 1))
    elif stype == "robust":
        return RobustScaler()
    elif stype == "quantile":
        return QuantileTransformer(
            n_quantiles=1000,
            random_state=seed,
            output_distribution="normal"
        )
    elif stype == "quantile-uniform":
        return QuantileTransformer(
            n_quantiles=1000,
            random_state=seed,
            output_distribution="uniform"
        )
    else:
        raise ValueError(
            "Preprocessing type not recognized. Select from standard, normalize, symmetric, robust, quantile, or quantile-uniform"
        )


def load_preprocessing(conf, seed=1000):
    if "scaler_x" not in conf["data"]:
        input_scaler = False
        logger.info("Loading input scaler: None")
    else:
        scaler_type = conf["data"]["scaler_x"]["type"]
        scaler_params = conf["data"]["scaler_x"]["params"]
        logger.info(f"Loading input scaler: {scaler_type}")
        input_scaler = load_preprocessor(scaler_type, scaler_params, seed)

    if "scaler_y" not in conf["data"]:
        output_scaler = False
        logger.info("Loading output scaler: None")
    else:
        scaler_type = conf["data"]["scaler_y"]["type"]
        scaler_params = conf["data"]["scaler_y"]["params"]
        logger.info(f"Loading output scaler: {scaler_type}")
        output_scaler = load_preprocessor(scaler_type, scaler_params, seed)

    return input_scaler, output_scaler
