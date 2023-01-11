import pathlib
from sys import float_info
from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.sleep_stage import SleepStageDataset, read_sleep_staging_labels
from transformiloop.src.utils.configs import initialize_config, validate_config

import unittest

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError("Error when initializing test config, check your config")

        self.subject_list = ['01-01-0001', '01-02-0001']
        self.MASS_dir = pathlib.Path(__file__).parents[1].resolve() / 'transformiloop' / 'dataset'

        # self.data = read_pretraining_dataset(self.MASS_dir / 'MASS_preds')

    def test_reader_sleep_staging_labels(self):
        labels = read_sleep_staging_labels(self.MASS_dir)
        print(labels)

    # Test pretraining dataset
    # def test_sleep_Staging_dataset(self):
    #     dataset = SleepStageDataset(self.subject_list, self.MASS_dir, self.data, self.config)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()