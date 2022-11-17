import pathlib
from sys import float_info
from transformiloop.src.data.pretraining import PretrainingDataset
from transformiloop.src.utils.configs import initialize_config, validate_config
import unittest
import torch
from torch.utils.data import DataLoader

MAX_ITER_TEST = 5


class TestPretrainingDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError(
                "Error when initializing test config, check your config")

        # Get the subject list for each dataset and the data for all of them
        self.dataset_path = pathlib.Path(__file__).parents[1].resolve(
        ) / 'transformiloop' / 'dataset' / 'pre_ds_test'

    # Test pretraining dataset
    def test_pretraining_dataset(self):
        pre_dataset = PretrainingDataset(self.dataset_path, self.config)

        pre_dl = DataLoader(
            pre_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        signal, gender, age, mask, reps = next(iter(pre_dl))
        self.assertEqual(signal.shape, torch.Size([self.config['batch_size'], self.config['seq_len'], self.config['window_size']]))       
        self.assertTrue(gender[0] == 0 or gender[0] == 1)
        self.assertTrue(age[0] >= 18 and age[0] <= 76) 
        self.assertEqual(mask.shape, torch.Size([self.config['batch_size'], self.config['seq_len']]))

        counter = torch.zeros(4)

        for i in range(MAX_ITER_TEST):
            _, _, _, mask, _ = next(iter(pre_dl))
            elems = []
            for elem in mask:
                elems.append(torch.bincount(elem, minlength=4))
            add = torch.stack(elems, dim=0).sum(dim=0)
            counter += add

        sampled = counter / sum(counter)
        for (samp, expected) in zip(sampled.tolist(), pre_dataset.mask_probs.tolist()):
            self.assertAlmostEqual(samp, expected, places=1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
