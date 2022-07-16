import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
import torch.fft as fft
import random
from transformiloop.src.data.augmentations import DataTransform_TD, DataTransform_FD


def get_subject_list(config):
    # Load all subject files
    all_subject = pd.read_csv(os.path.join(config['subjects_path'], "subject_sequence_full_big.txt"), header=None, delim_whitespace=True).to_numpy()
    test_subject = None
    p1_subject = pd.read_csv(os.path.join(config['subjects_path'], 'subject_sequence_p1_big.txt'), header=None, delim_whitespace=True).to_numpy()
    p2_subject = pd.read_csv(os.path.join(config['subjects_path'], 'subject_sequence_p2_big.txt'), header=None, delim_whitespace=True).to_numpy()

    # Get splits for train, validation and test
    train_subject_p1, validation_subject_p1 = train_test_split(p1_subject, train_size=0.8, random_state=config['seed'])
    test_subject_p1, validation_subject_p1 = train_test_split(validation_subject_p1, train_size=0.5, random_state=config['seed'])
    train_subject_p2, validation_subject_p2 = train_test_split(p2_subject, train_size=0.8, random_state=config['seed'])
    test_subject_p2, validation_subject_p2 = train_test_split(validation_subject_p2, train_size=0.5, random_state=config['seed'])

    # Get subject list depending on split
    train_subject = np.array([s for s in all_subject if s[0] in train_subject_p1[:, 0] or s[0] in train_subject_p2[:, 0]]).squeeze()
    test_subject = np.array([s for s in all_subject if s[0] in test_subject_p1[:, 0] or s[0] in test_subject_p2[:, 0]]).squeeze()
    validation_subject = np.array(
        [s for s in all_subject if s[0] in validation_subject_p1[:, 0] or s[0] in validation_subject_p2[:, 0]]).squeeze()

    print(f"Subjects in training : {train_subject[:, 0]}")
    print(f"Subjects in validation : {validation_subject[:, 0]}")
    print(f"Subjects in test : {test_subject[:, 0]}")
    
    return train_subject, validation_subject, test_subject

class FinetuneDataset(Dataset):
    def __init__(self, list_subject, config, augmentation_config=None):
        self.fe = config['fe']
        self.window_size = config['window_size']
        self.augmentation_config = augmentation_config
        self.data = pd.read_csv(config['data_path'], header=None).to_numpy()
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / (config['len_segment'] + 30 * self.fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.seq_len = config['seq_len']  # 1 means single sample / no sequence ?
        self.seq_stride = config['seq_stride']
        self.past_signal_len = self.seq_len * self.seq_stride
        self.threshold = config['threshold']

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here
        total_spindles = np.sum(self.data[3] > self.threshold)
        print(f"total number of spindles in this dataset : {total_spindles}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size - 1] >= 0, f"Bad index: {idx}."

        x_data = self.full_signal[idx - (self.past_signal_len - self.seq_stride):idx + self.window_size].unfold(0, self.window_size, self.seq_stride)
        label = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)
        x_data_f = fft.fft(x_data).abs() 

        aug1, aug1_f = torch.zeros(x_data.shape), torch.zeros(x_data_f.shape)
        if self.augmentation_config is not None:
            aug1 = DataTransform_TD(x_data.unsqueeze(0), self.augmentation_config).squeeze(1)
            aug1_f = DataTransform_FD(x_data_f.unsqueeze(0)).squeeze(1)

        return x_data, x_data_f, label, aug1, aug1_f

    def is_spindle(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > self.threshold) else False

def get_class_idxs(dataset, distribution_mode):
    """
    Directly outputs idx_true and idx_false arrays
    """
    length_dataset = len(dataset)

    nb_true = 0
    nb_false = 0

    idx_true = []
    idx_false = []

    for i in range(length_dataset):
        is_spindle = dataset.is_spindle(i)
        if is_spindle or distribution_mode == 1:
            nb_true += 1
            idx_true.append(i)
        else:
            nb_false += 1
            idx_false.append(i)

    assert len(dataset) == nb_true + nb_false, f"Bad length dataset"

    return np.array(idx_true), np.array(idx_false)


class RandomSampler(Sampler):
    """
    Samples elements randomly and evenly between the two classes.
    The sampling happens WITH replacement.
    __iter__ stops after an arbitrary number of iterations = batch_size_list * nb_batch
    Arguments:
      idx_true: np.array
      idx_false: np.array
      batch_size (int)
      nb_batch (int, optional): number of iteration before end of __iter__(), this defaults to len(data_source)
    """

    def __init__(self, idx_true, idx_false, config):
        self.idx_true = idx_true
        self.idx_false = idx_false
        self.nb_true = self.idx_true.size
        self.nb_false = self.idx_false.size
        self.length = config['training_batches'] * config['batch_size']

    def __iter__(self):
        global precision_validation_factor
        global recall_validation_factor
        cur_iter = 0
        proba = 0.5

        while cur_iter < self.length:
            cur_iter += 1
            sample_class = np.random.choice([0, 1], p=[1 - proba, proba])
            if sample_class:  # sample true
                idx_file = random.randint(0, self.nb_true - 1)
                idx_res = self.idx_true[idx_file]
            else:  # sample false
                idx_file = random.randint(0, self.nb_false - 1)
                idx_res = self.idx_false[idx_file]

            yield idx_res

    def __len__(self):
        return self.length

def get_dataloaders(config):
    subs_train, subs_val, subs_test = get_subject_list(config['MODA_data_config'])
    train_ds = FinetuneDataset(subs_train, config['MODA_data_config'], augmentation_config=config['augmentation_config'])
    val_ds = FinetuneDataset(subs_val, config['MODA_data_config'], augmentation_config=config['augmentation_config'])
    test_ds = FinetuneDataset(subs_test, config['MODA_data_config'], augmentation_config=config['augmentation_config'])

    idx_true, idx_false = get_class_idxs(train_ds, 0)

    train_sampler = RandomSampler(idx_true, idx_false, config['MODA_data_config'])

    train_dl = DataLoader(
        train_ds, 
        batch_size=config['MODA_data_config']['batch_size'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=config['MODA_data_config']['batch_size'],
        # sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    test_dl = DataLoader(
        test_ds, 
        batch_size=config['MODA_data_config']['batch_size'],
        # sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    return train_dl, val_dl, test_dl


    def __len__(self):
        return self.length