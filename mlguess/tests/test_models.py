import warnings
import keras
warnings.filterwarnings("ignore")
import yaml
import unittest
import numpy as np
from mlguess.keras.models import CategoricalDNN, EvidentialRegressorDNN
from mlguess.keras.losses import evidential_cat_loss, evidential_reg_loss
from keras.models import load_model
from mlguess.keras.callbacks import ReportEpoch

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

    def test_evidential_categorical_model(self):
        n_classes = 5
        epoch_var = keras.Variable(0)
        x_train = np.random.random(size=(10000, 10)).astype('float32')
        y_train = np.random.randint(size=(10000, n_classes), high=4, low=0).astype('float32')
        report_epoch_callback = ReportEpoch(epoch_var)
        model = CategoricalDNN(n_classes=n_classes)
        model.compile(loss=evidential_cat_loss(evi_coef=1.0, epoch_callback=report_epoch_callback), optimizer="adam")
        hist = model.fit(x_train, y_train, epochs=10, callbacks=report_epoch_callback)
        p_with_uncertainty = model.predict(x_train, return_uncertainties=True, batch_size=10000)
        p_without_uncertainty = model.predict(x_train, return_uncertainties=False, batch_size=10000)
        assert len(p_with_uncertainty) == 4
        assert p_with_uncertainty[0].shape[-1] == n_classes
        assert p_with_uncertainty[1].shape[-1] == 1
        assert p_with_uncertainty[2].shape[-1] == n_classes
        assert p_with_uncertainty[3].shape[-1] == n_classes
        assert p_without_uncertainty.shape[-1] == n_classes
        model.save("test_evi_categorical.keras")
        load_model("test_evi_categorical.keras")

    def test_evidential_regression_model(self):

        x_train = np.random.random(size=(10000, 10)).astype('float32')
        y_train = np.random.random(size=(10000, 1)).astype('float32')
        model = EvidentialRegressorDNN(hidden_layers=2)
        model.compile(loss=evidential_reg_loss(0.01), optimizer="adam")
        model.fit(x_train, y_train, epochs=10)
        p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
        p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
        assert p_with_uncertainty.shape[-1] == 3
        assert p_without_uncertainty.shape[-1] == 4
        model.save("test_evi_regression.keras")
        load_model("test_evi_regression.keras")

if __name__ == "__main__":
    unittest.main()