import os
import numpy as np
import pyedflib
import csv
from torch.utils.data import Dataset, DataLoader, Sampler

from transformiloop.src.data.pretraining import read_pretraining_dataset


def read_sleep_staging_labels(MASS_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    # Read the sleep_staging.csv file into a numpy array 
    sleep_staging_file = os.path.join(MASS_dir, 'sleep_staging.csv')


class SleepStageDataset(Dataset):
    def __init__(self, subjects, MASS_dir, data, labels, config):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        # list all files in the MASS directory
        files = os.listdir(MASS_dir)

        # Make sure that the files that we need are present
        assert 'sleep_staging.csv' in files
        assert 'MASS_preds' in files

        self.config = config

        self.full_signal = []
        self.full_labels = []

        for subject in subjects:
            assert subject in data.keys()
            signal = data[subject]['signal']
            # Get all the labels for the given subject
            label = []
            for lab in labels[subject]:
                label += [lab] * self.config['fe']
            
            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels += label
            self.full_signal += signal


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
