import pathlib
from sys import float_info
from tests.test_pretraining_dataset import MAX_ITER_TEST
from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.sleep_stage import SleepStageDataset, SleepStageSampler, read_sleep_staging_labels
from transformiloop.src.utils.configs import initialize_config, validate_config
import numpy as np
import unittest
import torch
from torch.utils.data import DataLoader


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError("Error when initializing test config, check your config")

        self.subject_list = ['01-05-0021', '01-05-0022']
        self.MASS_dir = pathlib.Path(__file__).parents[1].resolve() / 'transformiloop' / 'dataset'

        self.data = read_pretraining_dataset(self.MASS_dir / 'test_ds')

    def test_reader_sleep_staging_labels(self):
        labels = read_sleep_staging_labels(self.MASS_dir)
        self.assertEqual(len(labels.keys()), 160)

    # Test pretraining dataset
    def test_sleep_Staging_dataset(self):
        sleep_stages = read_sleep_staging_labels(self.MASS_dir)
        dataset = SleepStageDataset(self.subject_list, self.MASS_dir, self.data, sleep_stages, self.config)
        self.assertEqual(len(dataset), len(dataset.full_labels))
        
        # Get a random index
        random_index = np.random.randint(0, len(dataset))
        # Get the signal and the label
        signal, label = dataset[random_index]

        # Check that the signal is of the correct size
        self.assertEqual(signal.shape, torch.Size([self.config['seq_len'], self.config['window_size']]))

        sampler = SleepStageSampler(dataset)
        for _ in range(MAX_ITER_TEST):
            index = next(sampler.__iter__())
            self.assertNotEqual(dataset[index], torch.tensor([0, 0, 0, 0, 0]))

        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )

        for index, batch in enumerate(dataloader):
            # Check if th signal is of the right shape
            signal = batch[0]
            self.assertEqual(signal.shape, torch.Size([self.config['batch_size'], self.config['seq_len'], self.config['window_size']]))
            
            # Check if the labels are the right shape
            label = batch[1]
            self.assertEqual(label.shape, torch.Size([self.config['batch_size'], 5]))

            # Check that none of the labels is [0, 0, 0, 0, 0]
            self.assertFalse(torch.all(label == torch.tensor([0, 0, 0, 0, 0])))
            
            if index > MAX_ITER_TEST:
                break

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
