import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os
from mlguess.torch.checkpoint import TorchFSDPModel, FSDPOptimizerWrapper, TorchFSDPCheckpointIO


class TestFSDPCheckpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        if not torch.cuda.is_available():
            raise unittest.SkipTest("GPU is not available. Skipping torch.distributed tests.")

        dist.init_process_group("nccl", rank=0, world_size=1)

    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to("cuda")
        self.fsdp_model = TorchFSDPModel(self.model)
        self.optimizer = optim.SGD(self.fsdp_model.parameters(), lr=0.01)
        self.fsdp_optimizer = FSDPOptimizerWrapper(self.optimizer, self.fsdp_model)
        self.checkpoint_io = TorchFSDPCheckpointIO()
        self.rank = dist.get_rank()

    def test_save_and_load_model(self):
        # Save model
        self.checkpoint_io.save_unsharded_model(self.fsdp_model, "test_model_checkpoint.pth", rank=self.rank, gather_dtensor=True, use_safetensors=False)

        # Load model
        loaded_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to("cuda")
        fsdp_loaded_model = TorchFSDPModel(loaded_model)
        self.checkpoint_io.load_unsharded_model(fsdp_loaded_model, "test_model_checkpoint.pth")

        # Compare models
        original_state_dict = self.fsdp_model.state_dict()
        loaded_state_dict = fsdp_loaded_model.state_dict()
        for key in original_state_dict:
            if key != '_flat_param':
                self.assertTrue(torch.allclose(original_state_dict[key], loaded_state_dict[key]), f"Loaded model parameter {key} does not match original")

    def test_save_and_load_optimizer(self):
        # Save optimizer
        self.checkpoint_io.save_unsharded_optimizer(self.fsdp_optimizer, "test_optimizer_checkpoint.pth", rank=self.rank, gather_dtensor=True)

        # Load optimizer
        loaded_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to("cuda")
        fsdp_loaded_model = TorchFSDPModel(loaded_model)
        loaded_optimizer = optim.SGD(fsdp_loaded_model.parameters(), lr=0.01)
        fsdp_loaded_optimizer = FSDPOptimizerWrapper(loaded_optimizer, fsdp_loaded_model)
        self.checkpoint_io.load_unsharded_optimizer(fsdp_loaded_optimizer, "test_optimizer_checkpoint.pth")

        # Compare optimizers
        original_state_dict = self.fsdp_optimizer.optim.state_dict()
        loaded_state_dict = fsdp_loaded_optimizer.optim.state_dict()

        self.assertEqual(original_state_dict['param_groups'], loaded_state_dict['param_groups'], "Optimizer param_groups do not match")

        for key in original_state_dict['state']:
            self.assertTrue(key in loaded_state_dict['state'], f"Key {key} not found in loaded optimizer state")
            for param_key, param_value in original_state_dict['state'][key].items():
                if isinstance(param_value, torch.Tensor):
                    self.assertTrue(torch.allclose(param_value, loaded_state_dict['state'][key][param_key]),
                                    f"Optimizer state for key {key}, param {param_key} does not match")
                else:
                    self.assertEqual(param_value, loaded_state_dict['state'][key][param_key],
                                     f"Optimizer state for key {key}, param {param_key} does not match")

    def test_load_model_outside_fsdp(self):
        # Save model
        self.checkpoint_io.save_unsharded_model(self.fsdp_model, "test_model_checkpoint.pth", rank=self.rank, gather_dtensor=True, use_safetensors=False)

        # Load model outside FSDP context
        indy_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to("cuda")
        checkpoint = torch.load("test_model_checkpoint.pth")
        indy_model.load_state_dict(checkpoint)

        # Compare models
        fsdp_state_dict = self.fsdp_model.state_dict()
        indy_state_dict = indy_model.state_dict()

        for (fsdp_key, fsdp_value), (indy_key, indy_value) in zip(fsdp_state_dict.items(), indy_state_dict.items()):
            if '_flat_param' not in fsdp_key and '_flat_param' not in indy_key:
                self.assertTrue(torch.allclose(fsdp_value, indy_value), f"Parameter {fsdp_key} does not match {indy_key}")

    def tearDown(self):
        # Clean up checkpoint files
        if os.path.exists("test_model_checkpoint.pth"):
            os.remove("test_model_checkpoint.pth")
        if os.path.exists("test_optimizer_checkpoint.pth"):
            os.remove("test_optimizer_checkpoint.pth")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
