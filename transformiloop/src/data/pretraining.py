from copy import deepcopy
import logging
import os
import csv
import random
import pyedflib
from torch.utils.data import Dataset, Sampler
import numpy as np
import torch


def read_patient_info(dataset_path):
    """
    Read the patient info from a patient_info file and initialize it in a dictionary
    """
    patient_info_file = os.path.join(dataset_path, 'patient_info.csv')
    with open(patient_info_file, 'r') as patient_info_f:
        # Skip the header if present
        has_header = csv.Sniffer().has_header(patient_info_f.read(1024))
        patient_info_f.seek(0)  # Rewind.
        reader = csv.reader(patient_info_f)
        if has_header:
            next(reader)  # Skip header row.

        patient_info = {
            line[0]: {
                'age': int(line[1]),
                'gender': line[2]
            } for line in reader
        }
    return patient_info


def read_pretraining_dataset(dataset_path, patients_to_keep=None):
    """
    Load all dataset files into a dictionary to be ready for a Pytorch Dataset.
    Note that this will only read the first signal even if the EDF file contains more.
    """
    patient_info = read_patient_info(dataset_path)

    for patient_id in patient_info.keys():
        if patients_to_keep is not None and patient_id not in patients_to_keep:
            continue
        filename = os.path.join(dataset_path, patient_id + ".edf")
        try:
            with pyedflib.EdfReader(filename) as edf_file:
                patient_info[patient_id]['signal'] = edf_file.readSignal(0)
        except FileNotFoundError:
            logging.debug(f"Skipping file {filename} as it is not in dataset.")

    # Remove all patients whose signal is not in dataset
    dataset = {patient_id: patient_details for (patient_id, patient_details) in patient_info.items()
        if 'signal' in patient_info[patient_id].keys()}

    return dataset


class PretrainingDataset(Dataset):
    def __init__(self, dataset_path, config, device=None):
        self.device = device
        self.window_size = config['window_size']

        data = read_pretraining_dataset(dataset_path)

        def sort_by_gender_and_age(subject):
            res = 0
            assert data[subject]['age'] < 255, f"{data[subject]['age']} years is a bit old."
            if data[subject]['gender'] == 'M':
                res += 1000
            res += data[subject]['age']
            return res

        self.subjects = sorted(data.keys(), key=sort_by_gender_and_age)
        self.nb_subjects = len(self.subjects)

        logging.debug(f"DEBUG: {self.nb_subjects} subjects:")
        for subject in self.subjects:
            logging.debug(
                f"DEBUG: {subject}, {data[subject]['gender']}, {data[subject]['age']} yo")

        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + \
            self.window_size  # signal needed for one sample

        self.full_signal = []
        self.genders = []
        self.ages = []

        for subject in self.subjects:
            assert self.min_signal_len <= len(
                data[subject]['signal']), f"Signal {subject} is too short."
            data[subject]['signal'] = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            self.full_signal.append(data[subject]['signal'])
            gender = 1 if data[subject]['gender'] == 'M' else 0
            age = data[subject]['age']
            ones = torch.ones_like(data[subject]['signal'], dtype=torch.uint8)
            gender_tensor = ones * gender
            age_tensor = ones * age
            self.genders.append(gender_tensor)
            self.ages.append(age_tensor)
            del data[subject]  # we delete this as it is not needed anymore

        # all signals concatenated (float32)
        self.full_signal = torch.cat(self.full_signal)
        # all corresponding genders (uint8)
        self.genders = torch.cat(self.genders)
        self.ages = torch.cat(self.ages)  # all corresponding ages (uint8)
        assert len(self.full_signal) == len(self.genders) == len(self.ages)

        self.samplable_len = len(self.full_signal) - self.min_signal_len + 1

        # Masking probabilities
        prob_not_masked = 1 - config['ratio_masked']
        prob_masked = config['ratio_masked'] * (1 - (config['ratio_replaced'] + config['ratio_kept']))
        prob_replaced = config['ratio_masked'] * config['ratio_replaced']
        prob_kept = config['ratio_masked'] * config['ratio_kept']
        self.mask_probs = torch.tensor([prob_not_masked, prob_masked, prob_replaced, prob_kept])
        self.mask_cum_probs = self.mask_probs.cumsum(0)

    def __len__(self):
        return self.samplable_len

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."

        idx += self.past_signal_len

        # Get data
        x_data = self.full_signal[idx - self.past_signal_len:idx + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        # TODO: double-check
        x_gender = self.genders[idx + self.window_size - 1]
        x_age = self.ages[idx + self.window_size - 1]  # TODO: double-check
        
        # Get random mask from given probabilities:
        mask = torch.searchsorted(self.mask_cum_probs, torch.rand(self.seq_len))

        # Get the sequence for masked sequence modeling
        masked_seq = x_data.clone()
        for seq_idx, mask_token in enumerate(mask):
            # No mask or skip mask or MASK token (which is done later)
            if mask_token in [0, 1, 3]: 
                continue
            elif mask_token == 2:
                # Replace token with replacement
                random_idx = int(torch.randint(high=len(self)-self.window_size, size=(1, )))
                masked_seq[seq_idx] = self.full_signal[random_idx: random_idx+self.window_size]
            else:
                raise RuntimeError("Issue with masks, shouldn't get a value not in {0, 1, 2, 3}")

        return x_data, x_gender, x_age, mask, masked_seq

class ValidationSampler(Sampler):
    def __init__(self, data_source, dividing_factor):
        self.len_max = len(data_source)
        self.data = data_source
        self.dividing_factor = dividing_factor

    def __iter__(self):
        for idx in range(0, self.len_max, self.dividing_factor):
            yield random.randint(0, self.len_max - 1)

    def __len__(self):
        return self.len_max // self.dividing_factor
