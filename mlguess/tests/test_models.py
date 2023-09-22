import warnings
warnings.filterwarnings("ignore")

import yaml
import unittest
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from mlguess.keras.models import BaseRegressor as RegressorDNN
from mlguess.keras.models import GaussianRegressorDNN
from mlguess.keras.models import EvidentialRegressorDNN

class TestModels(unittest.TestCase):
    def setUp(self):
        # Load configurations for the models
        self.mlp_config = "config/surface_layer/mlp.yml"
        self.gaussian_config = "config/surface_layer/gaussian.yml"
        self.evidential_config = "config/surface_layer/evidential.yml"

        with open(self.mlp_config) as cf:
            self.mlp_conf = yaml.load(cf, Loader=yaml.FullLoader)
            
        with open(self.gaussian_config) as cf:
            self.gaussian_conf = yaml.load(cf, Loader=yaml.FullLoader)

        with open(self.evidential_config) as cf:
            self.evidential_conf = yaml.load(cf, Loader=yaml.FullLoader)

        # Instantiate and preprocess the data (as you did before)...
        data_file = "data/sample_cabauw_surface_layer.csv"
        self.data = pd.read_csv(data_file)
        self.data["day"] = self.data["Time"].apply(lambda x: str(x).split(" ")[0])

        # Initialize scalers and split data
        self.x_scaler, self.y_scaler = RobustScaler(), MinMaxScaler((0, 1))
        self.input_cols = self.mlp_conf["data"]["input_cols"]
        self.output_cols = ['friction_velocity:surface:m_s-1']

        self.flat_seed = 1000
        gsp = GroupShuffleSplit(n_splits=1, random_state=self.flat_seed, train_size=0.9)
        splits = list(gsp.split(self.data, groups=self.data["day"]))
        train_index, test_index = splits[0]
        self.train_data, self.test_data = self.data.iloc[train_index].copy(), self.data.iloc[test_index].copy()

        gsp = GroupShuffleSplit(n_splits=1, random_state=self.flat_seed, train_size=0.885)
        splits = list(gsp.split(self.train_data, groups=self.train_data["day"]))
        train_index, valid_index = splits[0]
        self.train_data, self.valid_data = self.train_data.iloc[train_index].copy(), self.train_data.iloc[valid_index].copy()

        x_train = self.x_scaler.fit_transform(self.train_data[self.input_cols])
        x_valid = self.x_scaler.transform(self.valid_data[self.input_cols])
        x_test = self.x_scaler.transform(self.test_data[self.input_cols])

        y_train = self.y_scaler.fit_transform(self.train_data[self.output_cols])
        y_valid = self.y_scaler.transform(self.valid_data[self.output_cols])
        y_test = self.y_scaler.transform(self.test_data[self.output_cols])

        self.data = (x_train, y_train, x_valid, y_valid, x_test, y_test)

    def test_mlp_model(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.data

        # Instantiate and build the MLP model
        mlp_model = RegressorDNN(**self.mlp_conf["model"])
        mlp_model.build_neural_network(x_train.shape[-1], y_train.shape[-1])

        # Test the MLP model here...
        # Example: mlp_model.fit(...) and assertions

    def test_gaussian_model(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.data

        # Instantiate and build the GaussianRegressorDNN model
        gaussian_model = GaussianRegressorDNN(**self.gaussian_conf["model"])
        gaussian_model.build_neural_network(x_train.shape[-1], y_train.shape[-1])

        # Test the GaussianRegressorDNN model here...
        # Example: gaussian_model.fit(...) and assertions

    def test_evidential_model(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.data

        # Instantiate and build the EvidentialRegressorDNN model
        evidential_model = EvidentialRegressorDNN(**self.evidential_conf["model"])
        evidential_model.build_neural_network(x_train.shape[-1], y_train.shape[-1])
        assert evidential_model.model.output.shape[1] == 4
        # Test the EvidentialRegressorDNN model here...
        # Example: evidential_model.fit(...) and assertions

if __name__ == "__main__":
    unittest.main()