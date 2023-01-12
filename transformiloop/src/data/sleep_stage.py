import os
import random
import numpy as np
import pyedflib
import csv
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from transformiloop.src.data.pretraining import read_pretraining_dataset


def read_sleep_staging_labels(MASS_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    # Read the sleep_staging.csv file 
    sleep_staging_file = os.path.join(MASS_dir, 'sleep_staging.csv')
    with open(sleep_staging_file, 'r') as f:
        reader = csv.reader(f)
        # Remove the header line from the information
        sleep_staging = list(reader)[1:]

    sleep_stages = {}
    for i in range(len(sleep_staging)):
        subject = sleep_staging[i][0]
        sleep_stages[subject] = [stage for stage in sleep_staging[i][1:] if stage != '']

    return sleep_stages
    

class SleepStageSampler(Sampler):
    def __init__(self, dataset):
        # Get the indices of all the '?' in the labels
        self.indices = [i for i in range(len(dataset.full_labels)) if dataset.full_labels[i] != '?']

        # Shuffle the indices
        random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


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

        # 
        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []

        for subject in subjects:
            assert subject in data.keys()
            signal = data[subject]['signal']
            # Get all the labels for the given subject
            label = []
            for lab in labels[subject]:
                label += [lab] * self.config['fe']
            
            # Add some '?' padding at the end to make sure the length of signal and label match
            label += ['?'] * (len(signal) - len(label))

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels += label
            self.full_signal.append(torch.tensor(signal.tolist()))
        
        self.full_signal = torch.cat(self.full_signal)

    def __getitem__(self, index):
        # Get the signal and label at the given index
        index += self.past_signal_len

        # Get data
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        label = self.full_labels[index + self.window_size - 1]

        # Convert the label to a one-hot encoding
        all_labels = ['1', '2', '3', 'R', 'W']
        label = torch.tensor([1 if label == l else 0 for l in all_labels])

        return signal, label

    def __len__(self):
        return len(self.full_signal)
