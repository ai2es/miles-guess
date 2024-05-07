import warnings
warnings.filterwarnings("ignore")
import unittest
import numpy as np
from mlguess.keras.models import CategoricalDNN, RegressorDNN
from keras.models import load_model

class TestModels(unittest.TestCase):

    def test_evidential_categorical_model(self):
        n_classes = 5
        x_train = np.random.random(size=(10000, 10)).astype('float32')
        y_train = np.random.randint(size=(10000, n_classes), high=4, low=0).astype('float32')
        model = CategoricalDNN(evidential=True,
                               activation="leaky_relu",
                               n_classes=n_classes,
                               n_inputs=10,
                               epochs=10,
                               annealing_coeff=1.5,
                               lr=0.0001)
        hist = model.fit(x_train, y_train)
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

    def test_categorical_model(self):
        n_classes = 5
        x_train = np.random.random(size=(10000, 10)).astype('float32')
        y_train = np.random.randint(size=(10000, n_classes), high=4, low=0).astype('float32')
        model = CategoricalDNN(evidential=False,
                               loss="categorical_crossentropy",
                               output_activation="softmax",
                               n_classes=n_classes,
                               n_inputs=10,
                               epochs=10,
                               lr=0.0001)
        hist = model.fit(x_train, y_train)
        print(model.get_config())
        print(model.summary())
        p = model.predict(x_train, return_uncertainties=False, batch_size=10000)
        assert p.shape[-1] == n_classes
        assert p.shape[0] == x_train.shape[0]
        model.save("test_categorical.keras")
        load_model("test_categorical.keras")

    def test_evidential_regression_model(self):

        for n_outputs in range(1, 3):
            x_train = np.random.random(size=(10000, 10)).astype('float32')
            y_train = np.random.random(size=(10000, n_outputs)).astype('float32')
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_outputs,
                                 evidential=True,
                                 epochs=10)
            model.fit(x_train, y_train)
            p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
            p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
            assert p_with_uncertainty.shape[-1] == 3 * n_outputs
            assert p_without_uncertainty.shape[-1] == 4 * n_outputs
            model.save("test_evi_regression.keras")
            load_model("test_evi_regression.keras")

    def test_Gaussian_uncertainty_model(self):

        for n_outputs in range(1, 3):
            x_train = np.random.random(size=(10000, 10)).astype('float32')
            y_train = np.random.random(size=(10000, n_outputs)).astype('float32')
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_outputs,
                                 evidential=False,
                                 uncertainty=True,
                                 epochs=10)
            model.fit(x_train, y_train)
            p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
            p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
            assert p_with_uncertainty.shape[-1] == 2 * n_outputs
            assert p_without_uncertainty.shape[-1] == 2 * n_outputs
            model.save("test_evi_regression.keras")
            load_model("test_evi_regression.keras")

    def test_standard_nn_regression(self):

        for n_outputs in range(1, 3):
            x_train = np.random.random(size=(10000, 10)).astype('float32')
            y_train = np.random.random(size=(10000, n_outputs)).astype('float32')
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_outputs,
                                 evidential=False,
                                 uncertainty=False,
                                 epochs=10)
            model.fit(x_train, y_train)
            p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
            assert p_without_uncertainty.shape[-1] == n_outputs
            model.save("test_evi_regression.keras")
            load_model("test_evi_regression.keras")

if __name__ == "__main__":
    unittest.main()