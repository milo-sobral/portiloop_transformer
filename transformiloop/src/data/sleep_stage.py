import os
import random
import numpy as np
import pyedflib
import csv
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from transformiloop.src.data.pretraining import read_pretraining_dataset


def get_dataloaders_sleep_stage(MASS_dir, ds_dir, config):
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    """
    # Read all the subjects available in the dataset
    labels = read_sleep_staging_labels(ds_dir) 

    # Divide the subjects into train and test sets
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]

    # Read the pretraining dataset
    data = read_pretraining_dataset(MASS_dir)

    # Create the train and test datasets
    train_dataset = SleepStageDataset(train_subjects, data, labels, config)
    test_dataset = SleepStageDataset(test_subjects, data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SleepStageSampler(train_dataset, config),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=SleepStageSampler(test_dataset, config),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


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
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.window_size = config['window_size']
        self.max_len = len(dataset) - self.dataset.past_signal_len - self.window_size

    def __iter__(self):
        while True:
            index = random.randint(0, self.max_len - 1)
            # Make sure that the label at the end of the window is not '?'
            label = self.dataset.full_labels[index + self.dataset.past_signal_len + self.window_size - 1]
            if label != SleepStageDataset.get_labels().index('?'):
                yield index

    def __len__(self):
        return len(self.indices)


class SleepStageDataset(Dataset):
    def __init__(self, subjects, data, labels, config):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

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
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue
            # assert subject in data.keys(), f"Subject {subject} not found in the pretraining dataset" 
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            # Get all the labels for the given subject
            label = []
            for lab in labels[subject]:
                label += [SleepStageDataset.get_labels().index(lab)] * self.config['fe']
            
            # Add some '?' padding at the end to make sure the length of signal and label match
            label += [SleepStageDataset.get_labels().index('?')] * (len(signal) - len(label))

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels.append(torch.tensor(label, dtype=torch.uint8))
            self.full_signal.append(signal)
            del data[subject], signal, label
        
        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

    @staticmethod
    def get_labels():
        return ['1', '2', '3', 'R', 'W', '?']

    def __getitem__(self, index):
        # Get the signal and label at the given index
        index += self.past_signal_len

        # Get data
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        label = self.full_labels[index + self.window_size - 1]

        assert label != 5, "Label is '?'"

        return signal, label.type(torch.LongTensor)

    def __len__(self):
        return len(self.full_signal)
