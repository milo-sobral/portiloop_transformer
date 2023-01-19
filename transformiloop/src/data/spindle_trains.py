import json
import os
import random
import numpy as np
import pyedflib
import csv
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from transformiloop.src.data.pretraining import read_pretraining_dataset
from torch.utils.data.sampler import WeightedRandomSampler


def get_dataloaders_spindle_trains(MASS_dir, ds_dir, config):
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    """
    # Read all the subjects available in the dataset
    labels = read_spindle_trains_labels(ds_dir) 

    # Divide the subjects into train and test sets
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]

    # Read the pretraining dataset
    data = read_pretraining_dataset(MASS_dir)

    # Create the train and test datasets
    train_dataset = SpindleTrainDataset(train_subjects, data, labels, config)
    test_dataset = SpindleTrainDataset(test_subjects, data, labels, config)

    num_classes = 4
    weights = [1/num_classes for _ in range(num_classes)]

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=WeightedRandomSampler(weights, len(train_dataset)),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=WeightedRandomSampler(weights, len(test_dataset)),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


def read_spindle_trains_labels(ds_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    spindle_trains_file = os.path.join(ds_dir, 'spindle_train_annotations.json')
    # Read the json file
    with open(spindle_trains_file, 'r') as f:
        labels = json.load(f)
    return labels


class SpindleTrainDataset(Dataset):
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

            # Helper function that checks if a given index is in a list of ranges
            def in_range_label(index, ranges_lower, ranges_upper, labels):
                assert len(ranges_lower) == len(ranges_upper), "Issue with ranges"
                assert len(ranges_lower) == len(labels), "Issue with labels"
                for i in range(len(ranges_lower)):
                    if ranges_lower[i] <= index <= ranges_upper[i]:
                        return labels[i]
                return 0

            # Get all the labels for the given subject
            label = []
            for i in range(len(signal)):
                label.append(in_range_label(i, labels[subject]['onsets'], labels[subject]['offsets'], labels[subject]['labels_num']))

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
        return ['non-spindle', 'isolated', 'first', 'train']

    def __getitem__(self, index):
        # Get the signal and label at the given index
        index += self.past_signal_len

        # Get data
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        label = self.full_labels[index + self.window_size - 1]

        return signal, label.type(torch.LongTensor)

    def __len__(self):
        return len(self.full_signal)
