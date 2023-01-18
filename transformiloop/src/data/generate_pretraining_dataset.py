import argparse
import logging
import numpy as np
import pyedflib 
import os
from pathlib import Path
from pyedflib.highlevel import write_edf_quick
import pandas as pd



def generate_sleep_staging_dataset(MASS_directory, output_directory, subset_format='SS{}_EDF'):
    """
    Generates a large CSV file containing all the sleep staging data available in the given MASS directory.
    """
    # Collect all subset directories
    subsets_dir_names = [subset_format.format(i) for i in range(1, 6)]
    ss_dirs = [os.path.join(MASS_directory, dir) for dir in os.listdir(MASS_directory) if dir in subsets_dir_names]

    # Collect all desired filenames in those directories
    subject_filenames = extract_dataset_filenames(ss_dirs, filename_contains="Base.edf")

    # create empty dictionary
    data = {}

    # Extract info from the desired files
    for filename in subject_filenames:
        # Open edf file
        with pyedflib.EdfReader(filename) as edf_file:
            # Get the annotations
            annotations = edf_file.readAnnotations()
            assert len(annotations) == 3, "Annotations must be a tuple of (onsets, durations, labels)"

            # extract the data for this subject
            subject_id = os.path.basename(filename)[:10]
            subject_data = extract_sleep_staging_data(annotations)

            # add to dataframe
            data[subject_id] = subject_data

    # Save the data dictionary to json
    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_csv(os.path.join(output_directory, "sleep_staging.csv"))


def extract_sleep_staging_data(annotations):
    """
    Extracts the sleep staging data from the annotations of a single subject.
    """
    curr_index = 0
    labels = []
    for annotation in np.stack(annotations).T:
        onset, duration, label = annotation
        # Check if we have some empty space
        onset = float(onset)
        duration = float(duration)
        if onset > curr_index:
            # Add empty labels
            if int(onset - curr_index) > 0:
                logging.info(f"adding empty labels {int(onset - curr_index)}")
                labels += ['?'] * int(onset - curr_index)
        # Add the label
        assert label in ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Sleep stage ?'], f"Unknown label {label}"
        label = label.strip('Sleep stage ')
        labels += [label if label != '4' else '3'] * int(duration)
        curr_index = onset + duration

    return np.array(labels)


def generate_dataset(MASS_directory, output_directory, subset_format='SS{}_EDF', prim_channels=["C3"], ref_channel="A2", fe_in=256, fe_out=250):
    """
    Generates the pretraining dataset from MASS in the specified output directory.
    Note that the names of the folders of each subset must be in the format 'SS{subset number}_EDF'.
    We also assume that the subject IDs are the first ten characters of each filename.
    """

    # Collect all subset directories
    subsets_dir_names = [subset_format.format(i) for i in range(1, 6)]
    ss_dirs = [os.path.join(MASS_directory, dir) for dir in os.listdir(MASS_directory) if dir in subsets_dir_names]

    # Collect all desired filenames in those directories
    subject_filenames = extract_dataset_filenames(ss_dirs)
    
    # Read the reference file
    reference_list = pd.read_csv(os.path.join(MASS_directory, "reference_list.txt"), delim_whitespace=True)
    ref_dict = reference_list.groupby(['subject']).apply(lambda x: x['channel'].tolist()).to_dict()

    # Extract desired signal from the desired EDF files and store them in other EDFs.
    for filename in subject_filenames:
        # Get subject ID and check if we must rereference
        subject_id = os.path.basename(filename)[:10]
        reref = None
        try:
            reref = ref_dict[subject_id][0] == 'C3-A2'
        except KeyError:
            print(f"Skipping subject {subject_id} as we do not have the referencing information yet.")
        # Read Signal and rereference if necessary
        ref_chann = ref_channel if reref else None
        signals = readEDF(filename, prim_channels, ref_chann)

        if signals is not None:
            # Write to a new EDF file in desired output directory
            subject_id = os.path.basename(filename)[:10]
            out_file = os.path.join(output_directory, subject_id + ".edf")

            # If the in frequency and out frequency are different, resample the signal
            if fe_in != fe_out:
                signals = [resample_signal(signal) for signal in signals]

            # Write to a new EDF file
            print(f"Writing new edf file {out_file}")
            write_edf_quick(out_file, signals, fe_out)


def resample_signal(signal, fe_in=256, fe_out=250):
    """ 
    Resamples the given signal from fe_in to fe_out
    """
    len_seconds = len(signal) / fe_in
    number_points_out = int(len_seconds * fe_out)
    x = np.linspace(0, len(signal), num=number_points_out)
    xp = np.linspace(0, len(signal), num=len(signal))
    assert len(xp) == len(signal)
    return np.interp(x, xp, signal)


def extract_dataset_filenames(directories, filename_contains="PSG.edf"):
    """
    Parse the given directories and return a list of all files that contains the given string. 
    """
    return [os.path.join(dir, file) for dir in directories for file in os.listdir(dir) if filename_contains in file]


def readEDF(filename, prim_channels, ref_channel=None):
    """
    Read an EDF file from MASS Dataset, extract the desired primary channels and return them.
    Uses the reference channel to rereference them.
    """
    signals = None
    try:
        with pyedflib.EdfReader(filename) as edf_file:
            # Read the signal labels
            signal_labels = edf_file.getSignalLabels()
            # Extract desired primary channels
            indices_prim = [i for i, text in enumerate(signal_labels) for prim_channel in prim_channels if prim_channel in text and 'EEG' in text]

            # Check that we have found the right primary channel (i.e. we have at least one)
            if len(indices_prim) < 1:
                raise AttributeError(f"No signals with labels {prim_channels} could be found in {filename}.")

            # Get all the desired signals
            signals = [edf_file.readSignal(index) for index in indices_prim]

            # If we have a referencing channel, use it to reference our signal
            if ref_channel is not None:
                # Extract its index
                index_ref = [i for i, text in enumerate(signal_labels) if ref_channel in text and 'EEG' in text]

                # Check that we only have one ref channel given
                if len(index_ref) != 1:
                    print(signal_labels)
                    raise AttributeError(f"Invalid reference channel {ref_channel} in {filename}.")

                # Load the reference signal and rereference all the previously loaded signals
                ref_signal = edf_file.readSignal(index_ref[0])
                for signal in signals:
                    assert len(ref_signal) == len(signal), f"Issue with rereferencing signal in {filename}"
                signals = [signal - ref_signal for signal in signals]
    except OSError:
        print("File " + filename + " ignored because of corruption")

    return signals


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser()
    # Mutually exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    # Create a boolean argument
    group.add_argument('-r', '--pretraining', action='store_true', help='Generate the pretraining dataset.')
    group.add_argument('-s', '--sleep_staging', action='store_true', help='Generate the Sleep Staging dataset.')
    group.add_argument('-a', '--all', action='store_true', help='Generate all datasets.')
    args = parser.parse_args()

    MASS_directory = os.path.abspath("/project/MASS")
    output_directory = os.path.abspath('transformiloop/dataset/MASS_preds')

    if args.pretraining or args.all:
        print("Generating pretraining dataset")
        generate_dataset(MASS_directory=MASS_directory, output_directory=output_directory)
    
    if args.sleep_staging or args.all:
        print("Generating sleep staging dataset")
        generate_sleep_staging_dataset(MASS_directory=MASS_directory, output_directory='transformiloop/dataset/')

