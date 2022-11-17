import pathlib
from sys import float_info
from transformiloop.src.data.spindle_detection import FinetuneDataset, RandomSampler, get_class_idxs, get_info_subject, get_subject_list, get_data
from transformiloop.src.utils.configs import initialize_config, validate_config
import unittest
import torch
from torch.utils.data import DataLoader


MAX_ITER_TEST = 100

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError("Error when initializing test config, check your config")
        
        # Get the subject list for each dataset and the data for all of them
        dataset_path = pathlib.Path(__file__).parents[1].resolve() / 'transformiloop' / 'dataset'
        self.subs_train, self.subs_val, self.subs_test = get_subject_list(self.config, dataset_path)

        # Use only one subject for each set
        self.subs_train = self.subs_train[:1]
        self.subs_val = self.subs_val[:1]
        self.subs_test = self.subs_test[:1]
        self.data = get_data(dataset_path)

    # Test pretraining dataset
    def test_pretraining_dataset(self):
        
        # Set the necessary parameters in the config
        self.config["pretraining"] = True
        self.config["modif_ratio"] = 0.5
        self.assertEqual(validate_config(self.config), True)

        # Let's start by loading a small dataset
        train_ds = FinetuneDataset(self.subs_train, self.config, self.data, self.config['full_transformer'], device=self.config['device'])
        val_ds = FinetuneDataset(self.subs_val, self.config, self.data, False, device=self.config['device'])
        
        # Get sampler and Dataloader for all sets 
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
        val_dl = DataLoader(
            val_ds, 
            batch_size=self.config['batch_size_validation'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

        # Check that the shape of all vectors is in fact what we want
        first_element_train, first_label_train = next(iter(train_dl))
        self.assertEqual(first_element_train.shape, torch.Size([self.config['batch_size'], self.config['seq_len'], self.config['window_size']]))
        self.assertEqual(first_label_train.shape, torch.Size([self.config['batch_size']]))
        first_element_val, first_label_val = next(iter(val_dl))
        self.assertEqual(first_element_val.shape, torch.Size([self.config['batch_size_validation'], self.config['seq_len'], self.config['window_size']]))
        self.assertEqual(first_label_val.shape, torch.Size([self.config['batch_size_validation']]))

        # Check that default modification works as intended
        # Gets the first index where the label is zero (no modifications have been made)
        index = (first_label_train == 0).nonzero()[0]
        modified = train_ds.default_modif(first_element_train[index])

        # Check that the proportion of modified is roughly what we want
        ones, total = 0, 0
        for idx, (_, label) in enumerate(train_dl):
            if idx > MAX_ITER_TEST:
                break
            ones += label.sum(dim=0)
            total += label.size(0)
        self.assertAlmostEqual(self.config['modif_ratio'], float(ones/total), places=1)        

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()