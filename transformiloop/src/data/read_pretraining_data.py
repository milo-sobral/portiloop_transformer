import os
import csv
import pyedflib
from torch.utils.data import Dataset
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


def read_pretraining_dataset(dataset_path):
    """
    Load all dataset files into a dictionary to be ready for a Pytorch Dataset.
    Note that this will only read the first signal even if the EDF file contains more.
    """
    patient_info = read_patient_info(dataset_path)
    for patient_id in patient_info.keys():
        filename = os.path.join(dataset_path, patient_id + ".edf")
        try:
            with pyedflib.EdfReader(filename) as edf_file:
                patient_info[patient_id]['signal'] = edf_file.readSignal(0)
        except FileNotFoundError:
            print(f"Skipping file {filename} as it is not in dataset.")
    
    return patient_info


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

        print(f"DEBUG: {self.nb_subjects} subjects:")
        for subject in self.subjects:
            print(f"DEBUG: {subject}, {data[subject]['gender']}, {data[subject]['age']} yo")

        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride  # signal needed before the last window
        self.min_signal_len = self.past_signal_len + self.window_size  # signal needed for one sample

        self.full_signal = []
        self.genders = []
        self.ages = []

        for subject in self.subjects:
            assert self.min_signal_len <= len(data[subject]['signal']), f"Signal {subject} is too short."
            data[subject]['signal'] = torch.tensor(data[subject]['signal'], dtype=torch.float)
            self.full_signal.append(data[subject]['signal'])
            gender = 1 if data[subject]['gender'] == 'M' else 0
            age = data[subject]['age']
            ones = torch.ones_like(data[subject]['signal'], dtype=torch.uint8)
            gender_tensor = ones * gender
            age_tensor = ones * age
            self.genders.append(gender_tensor)
            self.ages.append(age_tensor)
            del data[subject]  # we delete this as it is not needed anymore

        self.full_signal = torch.cat(self.full_signal)  # all signals concatenated (float32)
        self.genders = torch.cat(self.genders)  # all corresponding genders (uint8)
        self.ages = torch.cat(self.ages)  # all corresponding ages (uint8)
        assert len(self.full_signal) == len(self.genders) == len(self.ages)

        self.samplable_len = len(self.full_signal) - self.min_signal_len + 1

    def __len__(self):
        return self.samplable_len

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."

        idx += self.past_signal_len

        # Get data
        x_data = self.full_signal[idx - self.past_signal_len:idx + self.window_size].unfold(0, self.window_size, self.seq_stride)  # TODO: double-check
        x_gender = self.genders[idx + self.window_size - 1]  # TODO: double-check
        x_age = self.ages[idx + self.window_size - 1]  # TODO: double-check

        return x_data, x_gender, x_age
