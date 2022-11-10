import os
import csv
import pyedflib


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

