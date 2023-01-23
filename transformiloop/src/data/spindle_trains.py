from copy import deepcopy
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

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=EquiRandomSampler(),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=EquiRandomSampler(),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


class EquiRandomSampler(Sampler):
    def __init__(self, ratio=0.5):
        self.ratio_spindles = ratio

    def __iter__(self):
        # Check that self.full_labels and self.dataset.full_labels are the same
        while True:
            # Get a random between 0 and 1
            next_label = random.random()
            if next_label < self.ratio_spindles:
                yield 0
            else:
                yield 1
            
    def __len__(self):
        return len(self.dataset)


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
        self.min_signal_len = self.past_signal_len + self.window_size

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []
        self.spindle_labels = []

        accumulator = 0
        for subject in subjects:
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue

            # Get the signal for the given subject
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)

            # Get all the labels for the given subject
            label = torch.zeros_like(signal, dtype=torch.uint8)
            spindle_label = []
            for (onset, offset, l) in zip(labels[subject]['onsets'], labels[subject]['offsets'], labels[subject]['labels_num']):
                
                label[onset:offset] = l
                # Make a separate list with the indexes of all the spindle labels so that sampling is easier
                to_add = list(range(accumulator + onset, accumulator + offset))
                spindle_label += to_add

            # increment the accumulator
            accumulator += len(signal)

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)
            # # Make sure that there arent too many spindles labeled
            # assert sum(torch.where(label != 0, 1, 0)) == len(spindle_label)
            # assert sum(torch.where(label == 0, 1, 0)) + len(spindle_label) == len(signal), f"Too many spindles labeled for subject {subject}"

            # Add to full signal and full label
            self.full_labels.append(label)
            self.full_signal.append(signal)
            self.spindle_labels += spindle_label
            del data[subject], signal, label
        
        # Concatenate the full signal and the full labels into one continuous tensor
        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

        # # Make sure that all indices in spindle_labels are indeed spindles in the signal
        # assert all([self.full_labels[index_label] != 0 for index_label in self.spindle_labels]), "Issue with the spindle labels"
        # # Make sure that the signal and the labels are the same length
        # assert len(self.full_signal) == len(self.full_labels), "Issue with the data and the labels"
        # # Make sure that the last spindle in the spindle_labels is in fact in the signal
        # assert all([index_label < len(self.full_labels) for index_label in self.spindle_labels]), "Issue with the spindle labels"


    @staticmethod
    def get_labels():
        return ['non-spindle', 'isolated', 'first', 'train']

    def __getitem__(self, index):
        # Get the signal and label at the given index
        # If index is 0, sample from the spindle labels list
        if index == 0:
            index = random.choice(self.spindle_labels)
            assert index in self.spindle_labels, "Spindle index not found in list"
            index = index - self.past_signal_len - self.window_size + 1
        else:
            # Sample from the rest of the signal
            index = random.randint(0, len(self.full_signal) - self.min_signal_len - self.window_size)

        # Get data
        index = index + self.past_signal_len
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)
        label = self.full_labels[index + self.window_size - 1]

        # Make sure that the last index of the signal is the same as the label
        assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        label = label.type(torch.LongTensor)

        return signal, label

    def __len__(self):
        return len(self.full_signal) - self.window_size
