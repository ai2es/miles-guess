import unittest
import torch
import numpy as np
from mlguess.torch.models import DNN  # Make sure to import your DNN class


class TestDNNModel(unittest.TestCase):

    def test_evidential_regression_model(self):
        for n_outputs in range(1, 3):
            input_size = 10
            x_train = torch.rand(10000, input_size)
            y_train = torch.rand(10000, n_outputs)

            model1 = DNN(input_size=input_size,
                        output_size=n_outputs,
                        layer_size=[100, 50],
                        dr=[0.2, 0.2],
                        batch_norm=True,
                        lng=True)  # Using LinearNormalGamma for evidential regression

            model2 = DNN(input_size=input_size,
                        output_size=n_outputs,
                        layer_size=[100, 50],
                        dr=[0.2, 0.2],
                        batch_norm=True,
                        lng=False)  # Using LinearNormalGamma for evidential regression

            # Train the model (you might need to implement a training loop)
            # For simplicity, we're skipping the training step here

            with torch.no_grad():
                p_with_uncertainty = model1.predict(x_train, return_uncertainties=True)
                p_without_uncertainty = model2.predict(x_train, return_uncertainties=False)

            self.assertEqual(len(p_with_uncertainty), 4)  # mu, aleatoric, epistemic, total
            self.assertEqual(len(p_with_uncertainty), 4)
            self.assertEqual(p_without_uncertainty.shape[-1], n_outputs)

            # Save and load model
            torch.save(model1.state_dict(), "test_evi_regression.pt")
            loaded_model = DNN(
                input_size=input_size,
                output_size=n_outputs,
                layer_size=[100, 50],
                dr=[0.2, 0.2],
                batch_norm=True,
                lng=True)  # Using LinearNormalGamma for evidential regression
            loaded_model.load_state_dict(torch.load("test_evi_regression.pt"))

    def test_standard_nn_regression(self):
        for n_outputs in range(1, 3):
            input_size = 10
            x_train = torch.rand(10000, input_size)
            y_train = torch.rand(10000, n_outputs)

            model = DNN(input_size=input_size,
                        output_size=n_outputs,
                        layer_size=[100, 50],
                        dr=[0.2, 0.2],
                        batch_norm=True,
                        lng=False)  # Standard neural network

            # Train the model (you might need to implement a training loop)
            # For simplicity, we're skipping the training step here

            with torch.no_grad():
                p_without_uncertainty = model.predict(x_train, return_uncertainties=False)

            self.assertEqual(p_without_uncertainty.shape[-1], n_outputs)

            # Save and load model
            torch.save(model.state_dict(), "test_standard_regression.pt")
            loaded_model = DNN(
                input_size=input_size,
                output_size=n_outputs,
                layer_size=[100, 50],
                dr=[0.2, 0.2],
                batch_norm=True,
                lng=False
            )  # Standard neural network
            loaded_model.load_state_dict(torch.load("test_standard_regression.pt"))

    def test_monte_carlo_dropout(self):
        input_size = 10
        output_size = 2
        x_train = torch.rand(1000, input_size)

        model = DNN(input_size=input_size,
                    output_size=output_size,
                    layer_size=[100, 50],
                    dr=[0.2, 0.2],
                    batch_norm=True,
                    lng=False)

        pred_probs, aleatoric, epistemic, entropy, mutual_info = model.predict_dropout(x_train)

        self.assertEqual(pred_probs.shape, (1000, output_size))
        self.assertEqual(aleatoric.shape, (1000, output_size))
        self.assertEqual(epistemic.shape, (1000, output_size))
        self.assertEqual(entropy.shape, (1000,))
        self.assertEqual(mutual_info.shape, (1000,))


if __name__ == "__main__":
    unittest.main()
