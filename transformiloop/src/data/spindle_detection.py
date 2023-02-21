from copy import deepcopy
import logging
from pathlib import Path
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
import torch.fft as fft
import random
from transformiloop.src.data.augmentations import DataTransform_TD, DataTransform_FD

DATASET_FILE = 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt'

def get_subject_list(config, dataset_path):
    # Load all subject files
    all_subject = pd.read_csv(os.path.join(dataset_path, "subject_sequence_full_big.txt"), header=None, delim_whitespace=True).to_numpy()
    test_subject = None
    p1_subject = pd.read_csv(os.path.join(dataset_path, 'subject_sequence_p1_big.txt'), header=None, delim_whitespace=True).to_numpy()
    p2_subject = pd.read_csv(os.path.join(dataset_path, 'subject_sequence_p2_big.txt'), header=None, delim_whitespace=True).to_numpy()

    # Get splits for train, validation and test
    train_subject_p1, validation_subject_p1 = train_test_split(p1_subject, train_size=0.8, random_state=None)
    test_subject_p1, validation_subject_p1 = train_test_split(validation_subject_p1, train_size=0.5, random_state=None)
    train_subject_p2, validation_subject_p2 = train_test_split(p2_subject, train_size=0.8, random_state=None)
    test_subject_p2, validation_subject_p2 = train_test_split(validation_subject_p2, train_size=0.5, random_state=None)

    # Get subject list depending on split
    train_subject = np.array([s for s in all_subject if s[0] in train_subject_p1[:, 0] or s[0] in train_subject_p2[:, 0]]).squeeze()
    test_subject = np.array([s for s in all_subject if s[0] in test_subject_p1[:, 0] or s[0] in test_subject_p2[:, 0]]).squeeze()
    validation_subject = np.array(
        [s for s in all_subject if s[0] in validation_subject_p1[:, 0] or s[0] in validation_subject_p2[:, 0]]).squeeze()

    print(f"Subjects in training : {train_subject[:, 0]}")
    print(f"Subjects in validation : {validation_subject[:, 0]}")
    print(f"Subjects in test : {test_subject[:, 0]}")
    
    return train_subject, validation_subject, test_subject


def get_data(dataset_path):
    start = time.time()
    data = pd.read_csv(os.path.join(dataset_path, DATASET_FILE), header=None).to_numpy()
    end = time.time()
    logging.info(f"Loaded data in {(end-start)} seconds...")
    return data

class FinetuneDataset(Dataset):
    def __init__(self, list_subject, config, data, history, augmentation_config=None, device=None, signal_modif=None):
        self.fe = config['fe']
        self.device = device
        self.window_size = config['window_size']
        self.augmentation_config = augmentation_config
        self.data = data
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / (config['len_segment'] + 30 * self.fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_labels = torch.tensor(self.data[3], dtype=torch.float)
        self.seq_len = 1 if config['full_transformer'] and not history else config['seq_len']  # want a single sample if full transformer and not training (aka validating), else we use seq len
        self.seq_stride = config['seq_stride']
        self.past_signal_len = self.seq_len * self.seq_stride
        self.threshold = config['data_threshold']
        self.label_history = history

        # Check if we are pretrining the model
        self.pretraining = config['pretraining']
        self.modif_ratio = config['modif_ratio']
        self.signal_modif = signal_modif

        # list of indices that can be sampled:
        if self.label_history:
            self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                            if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                    or self.data[3][idx - (self.past_signal_len - self.seq_stride) + self.window_size - 1] < 0  # and not beginning in an unlabeled zone
                                    or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here
        else:
            self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)
                            # all possible idxs in the dataset
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

        # Get data 
        x_data = self.full_signal[idx - (self.past_signal_len - self.seq_stride):idx + self.window_size].unfold(0, self.window_size, self.seq_stride)

        if self.pretraining:
            if random.uniform(0, 1) < self.modif_ratio:
                x_data = self.signal_modif(x_data) if self.signal_modif is not None else self.default_modif(x_data)
                label = torch.tensor(1, dtype=torch.float)
            else:
                label = torch.tensor(0, dtype=torch.float)
        else:
            # Get label for the spindle recognition task
            label_unique = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)
            if self.label_history:
                # Get the label history if we want to learn from that as well.
                label_history = self.full_labels[idx - (self.past_signal_len - self.seq_stride) + self.window_size - 1:idx + self.window_size].unfold(0, 1, self.seq_stride)
                assert len(label_history) == len(x_data), f"len(label):{len(label_history)} != len(x_data):{len(x_data)}"
                assert -1 not in label_history, f"invalid label: {label_history}"
                assert label_unique == label_history[-1], f"bad label: {label_unique} != {label_history[-1]}"
            label = label_history if self.label_history else label_unique

        assert label in [0, 1], f"Invalid label: {label}"
        label = label.type(torch.LongTensor)
        return x_data, label
        

    def is_spindle(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > self.threshold) else False
    
    def default_modif(self, signal):
        # Get one random sequence
        modified_index = random.randint(0, signal.size(0)-1)
        new_sig = deepcopy(signal)
        new_sig[modified_index] = -signal[modified_index]
        return new_sig

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
        self.length = config['batches_per_epoch'] * config['batch_size']

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
            
            # print('Sampled at index {}'.format(idx_res))
            yield idx_res

    def __len__(self):
        return self.length


class ValidationSamplerSimple(Sampler):
    def __init__(self, data_source, dividing_factor):
        self.len_max = len(data_source)
        self.data = data_source
        self.dividing_factor = dividing_factor

    def __iter__(self):
        for idx in range(0, self.len_max, self.dividing_factor):
            yield idx

    def __len__(self):
        return self.len_max // self.dividing_factor


class SignalDataset(Dataset):
    def __init__(self, filename, path, window_size, fe, seq_len, seq_stride, list_subject, len_segment):
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename

        self.data = pd.read_csv(self.path_file, header=None).to_numpy()
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / (len_segment + 30 * fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_envelope = torch.tensor(self.data[1], dtype=torch.float)
        self.seq_len = seq_len  # 1 means single sample / no sequence ?
        self.idx_stride = seq_stride
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here
        total_spindles = np.sum(self.data[3] > 0.2)
        logging.debug(f"total number of spindles in this dataset : {total_spindles}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size - 1] >= 0, f"Bad index: {idx}."

        signal_seq = self.full_signal[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)
        envelope_seq = self.full_envelope[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)

        ratio_pf = torch.tensor(self.data[2][idx + self.window_size - 1], dtype=torch.float)
        label = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)

        return signal_seq, label

    def is_spindle(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > 0.2) else False


class ValidationSampler(Sampler):
    """
    network_stride (int >= 1, default: 1): divides the size of the dataset (and of the batch) by striding further than 1
    """

    def __init__(self, seq_stride, nb_segment, network_stride):
        network_stride = int(network_stride)
        assert network_stride >= 1
        self.network_stride = network_stride
        self.seq_stride = seq_stride
        self.nb_segment = nb_segment
        self.len_segment = 115 * 250  # 115 seconds x 250 Hz

    def __iter__(self):
        random.seed()
        batches_per_segment = self.len_segment // self.seq_stride  # len sequence = 115 s + add the 15 first s?
        cursor_segment = 0
        while cursor_segment < batches_per_segment:
            for i in range(self.nb_segment):
                for j in range(0, (self.seq_stride // self.network_stride) * self.network_stride, self.network_stride):
                    cur_idx = i * self.len_segment + j + cursor_segment * self.seq_stride
                    # print(f"i:{i}, j:{j}, self.len_segment:{self.len_segment}, cursor_batch:{cursor_batch}, self.seq_stride:{self.seq_stride}, cur_idx:{cur_idx}")
                    yield cur_idx
            cursor_segment += 1

    def __len__(self):
        assert False
        # return len(self.data)
        # return len(self.data_source)


def get_info_subject(subjects, config):
    nb_segment = len(np.hstack([range(int(s[1]), int(s[2])) for s in subjects]))
    batch_size = len(list(range(0, (config['seq_stride'] // config['network_stride']) * config['network_stride'], config['network_stride']))) * nb_segment
    return nb_segment, batch_size


def get_dataloaders(config, dataset_path):
    subs_train, subs_val, subs_test = get_subject_list(config, dataset_path)
    # # Use only one subject for each set
    # subs_train = subs_train[:1]
    # subs_val = subs_val[:1]
    # subs_test = subs_test[:1]
    data = get_data(dataset_path)

    train_ds = FinetuneDataset(subs_train, config, data, config['full_transformer'], augmentation_config=None, device=config['device'])
    val_ds = FinetuneDataset(subs_val, config, data, False, augmentation_config=None, device=config['device'])
    test_ds = FinetuneDataset(subs_test, config, data, False, augmentation_config=None, device=config['device'])

    idx_true, idx_false = get_class_idxs(train_ds, 0)

    train_sampler = RandomSampler(idx_true, idx_false, config)

    nb_segment_val, batch_size_val = get_info_subject(subs_val, config)
    nb_segment_test, batch_size_test = get_info_subject(subs_test, config)
    config['batch_size_validation'] = batch_size_val
    config['batch_size_test'] = batch_size_test

    print(f"Batch Size validation: {batch_size_val}")
    print(f"Batch Size test: {batch_size_test}")

    # if config['full_transformer']:
    val_sampler = ValidationSampler(config['seq_stride'], nb_segment_val, config['network_stride'])
    test_sampler = ValidationSampler(config['seq_stride'], nb_segment_test, config['network_stride'])
    # else:
    #     val_sampler = ValidationSamplerSimple(val_ds, config['network_stride'])
    #     test_sampler = ValidationSamplerSimple(test_ds, config['network_stride'])
    
    if config['pretraining']:
        train_dl = DataLoader(
            train_ds, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
    else:
        train_dl = DataLoader(
            train_ds, 
            batch_size=config['batch_size'],
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
        
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size_val,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        shuffle=False)

    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size_test,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=True)

    return train_dl, val_dl, test_dl
